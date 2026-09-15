package com.swift.guard

import android.app.*
import android.content.Context
import android.content.Intent
import android.graphics.Color
import android.graphics.PixelFormat
import android.graphics.drawable.GradientDrawable
import android.net.Uri
import android.os.Build
import android.os.IBinder
import android.provider.Settings
import android.util.Log
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.WindowManager
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import androidx.core.app.NotificationCompat

class OverlayService : Service(), SwiftTelemetryClient.TelemetryListener {

    companion object {
        const val ACTION_SHOW_POPUP = "com.swift.guard.SHOW_POPUP"
        const val ACTION_HIDE_POPUP = "com.swift.guard.HIDE_POPUP"
        const val CHANNEL_ID = "SwiftGuardOverlayChannel"
        const val NOTIFICATION_ID = 2001
        const val BRIDGE_NUMBER = "+12049000957"
    }

    private var windowManager: WindowManager? = null
    private var overlayView: View? = null
    private var telemetryClient: SwiftTelemetryClient? = null
    private var activeCallSid: String = ""
    private var currentCallerNumber: String = "Active Call"
    private var sessionStartTime: Long = 0L
    private var maxSpiRecorded: Double = 0.0
    private var highestStatusRecorded: String = "authentic"

    // UI elements on active widget
    private var riskTextView: TextView? = null
    private var statusTextView: TextView? = null
    private var severButton: Button? = null
    private var widgetContainer: LinearLayout? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        windowManager = getSystemService(Context.WINDOW_SERVICE) as WindowManager
        createNotificationChannel()
        startForeground(NOTIFICATION_ID, createNotification("SWIFT Deepfake Guard Active"))
        val prefs = getSharedPreferences("swift_prefs", Context.MODE_PRIVATE)
        val activeServerUrl = prefs.getString("server_url", "https://berna-uninfused-sherron.ngrok-free.dev") ?: "https://berna-uninfused-sherron.ngrok-free.dev"
        telemetryClient = SwiftTelemetryClient(serverUrl = activeServerUrl, listener = this)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val action = intent?.action
        val callerNumber = intent?.getStringExtra("callerNumber") ?: "Active Call"
        currentCallerNumber = callerNumber

        if (action == ACTION_SHOW_POPUP) {
            if (Build.VERSION.SDK_INT < Build.VERSION_CODES.M || Settings.canDrawOverlays(this)) {
                postToMain {
                    showInitialPrompt(callerNumber)
                }
            } else {
                Log.w("SWIFT_Overlay", "Cannot draw overlays: permission denied")
            }
        } else if (action == ACTION_HIDE_POPUP) {
            cleanupAndStop()
        }

        return START_STICKY
    }

    private fun showInitialPrompt(callerNumber: String) {
        removeOverlay()

        val layoutParams = WindowManager.LayoutParams(
            WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.WRAP_CONTENT,
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O)
                WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY
            else
                @Suppress("DEPRECATION") WindowManager.LayoutParams.TYPE_PHONE,
            WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON,
            PixelFormat.TRANSLUCENT
        ).apply {
            gravity = Gravity.TOP or Gravity.CENTER_HORIZONTAL
            y = 120
        }

        // White card with subtle border
        val container = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(48, 40, 48, 32)
            val bg = GradientDrawable().apply {
                setColor(Color.WHITE)
                cornerRadius = 32f
                setStroke(2, Color.parseColor("#E0E0E0"))
            }
            background = bg
            elevation = 32f
        }

        // Brand label
        val brand = TextView(this).apply {
            text = "S W I F T"
            setTextColor(Color.BLACK)
            textSize = 11f
            typeface = android.graphics.Typeface.DEFAULT_BOLD
            letterSpacing = 0.25f
            gravity = Gravity.CENTER
        }
        container.addView(brand)

        // Thin separator — layoutParams set OUTSIDE apply to avoid shadowing
        val sep1 = View(this).apply {
            setBackgroundColor(Color.parseColor("#F0F0F0"))
        }
        sep1.layoutParams = LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, 1
        ).also { it.setMargins(0, 16, 0, 20) }
        container.addView(sep1)

        val callerLabel = TextView(this).apply {
            text = "INCOMING CALL"
            setTextColor(Color.parseColor("#888888"))
            textSize = 9f
            letterSpacing = 0.15f
            gravity = Gravity.CENTER
            setPadding(0, 0, 0, 6)
        }
        container.addView(callerLabel)

        val callerText = TextView(this).apply {
            text = callerNumber
            setTextColor(Color.BLACK)
            textSize = 16f
            typeface = android.graphics.Typeface.DEFAULT_BOLD
            gravity = Gravity.CENTER
            setPadding(0, 0, 0, 8)
        }
        container.addView(callerText)

        val scanLabel = TextView(this).apply {
            text = "Activate AI voice analysis?"
            setTextColor(Color.parseColor("#888888"))
            textSize = 12f
            gravity = Gravity.CENTER
            setPadding(0, 0, 0, 24)
        }
        container.addView(scanLabel)

        // Capture WM params before entering Button.apply scope (avoids shadowing by View.layoutParams)
        val wlp = layoutParams
        val protectBtn = Button(this).apply {
            text = "PROTECT CALL  →"
            setTextColor(Color.WHITE)
            textSize = 12f
            typeface = android.graphics.Typeface.DEFAULT_BOLD
            letterSpacing = 0.08f
            background = GradientDrawable().apply {
                setColor(Color.BLACK)
                cornerRadius = 8f
            }
            setPadding(0, 32, 0, 32)
            setOnClickListener { activateLiveProtection(wlp) }
        }
        protectBtn.layoutParams = LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT,
            LinearLayout.LayoutParams.WRAP_CONTENT
        )
        container.addView(protectBtn)

        // Ghost dismiss button
        val dismissBtn = Button(this).apply {
            text = "MAYBE LATER"
            setTextColor(Color.parseColor("#888888"))
            textSize = 11f
            letterSpacing = 0.08f
            background = GradientDrawable().apply { setColor(Color.TRANSPARENT) }
            setPadding(0, 16, 0, 16)
            setOnClickListener { cleanupAndStop() }
        }
        dismissBtn.layoutParams = LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT,
            LinearLayout.LayoutParams.WRAP_CONTENT
        ).also { it.setMargins(0, 8, 0, 0) }
        container.addView(dismissBtn)

        setupDrag(container, layoutParams)
        overlayView = container
        try {
            windowManager?.addView(overlayView, layoutParams)
        } catch (e: Exception) {
            Log.e("SWIFT_Overlay", "Failed to add initial overlay: ${e.message}")
        }
    }


    private fun activateLiveProtection(params: WindowManager.LayoutParams) {
        removeOverlay()

        // White card, minimal border
        val container = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(40, 32, 40, 28)
            val bg = GradientDrawable().apply {
                setColor(Color.WHITE)
                cornerRadius = 32f
                setStroke(2, Color.parseColor("#E0E0E0"))
            }
            background = bg
            elevation = 32f
        }
        widgetContainer = container

        // Brand + connecting row
        val brandRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        val brandLabel = TextView(this).apply {
            text = "S W I F T"
            setTextColor(Color.BLACK)
            textSize = 10f
            typeface = android.graphics.Typeface.DEFAULT_BOLD
            letterSpacing = 0.2f
        }
        brandLabel.layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        statusTextView = TextView(this).apply {
            text = "CONNECTING..."
            setTextColor(Color.parseColor("#888888"))
            textSize = 9f
            letterSpacing = 0.1f
        }
        brandRow.addView(brandLabel)
        brandRow.addView(statusTextView)
        container.addView(brandRow)

        // Divider
        val sep = View(this).apply {
            setBackgroundColor(Color.parseColor("#F0F0F0"))
        }
        sep.layoutParams = LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, 1
        ).also { it.setMargins(0, 14, 0, 18) }
        container.addView(sep)

        // SPI label
        val spiLabel = TextView(this).apply {
            text = "SPI AI"
            setTextColor(Color.parseColor("#888888"))
            textSize = 9f
            letterSpacing = 0.15f
            gravity = Gravity.CENTER
        }
        container.addView(spiLabel)

        // Large SPI percentage — center stage
        riskTextView = TextView(this).apply {
            text = "0%"
            setTextColor(Color.BLACK)
            textSize = 52f
            typeface = android.graphics.Typeface.DEFAULT_BOLD
            gravity = Gravity.CENTER
            setPadding(0, 4, 0, 4)
        }
        container.addView(riskTextView)

        // Sever button — full width black, hidden until threat
        severButton = Button(this).apply {
            text = "SEVER CALL  →"
            setTextColor(Color.WHITE)
            textSize = 11f
            typeface = android.graphics.Typeface.DEFAULT_BOLD
            letterSpacing = 0.08f
            visibility = View.GONE
            background = GradientDrawable().apply {
                setColor(Color.BLACK)
                cornerRadius = 8f
            }
            setPadding(0, 28, 0, 28)
            setOnClickListener {
                telemetryClient?.severCall(activeCallSid) { success ->
                    postToMain {
                        riskTextView?.text = if (success) "SEVERED" else "FAILED"
                        severButton?.visibility = View.GONE
                    }
                }
            }
        }
        severButton?.layoutParams = LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT,
            LinearLayout.LayoutParams.WRAP_CONTENT
        ).also { it.setMargins(0, 16, 0, 0) }
        container.addView(severButton)

        // Close — ghost text button
        val closeBtn = Button(this).apply {
            text = "CLOSE GUARD"
            setTextColor(Color.parseColor("#AAAAAA"))
            textSize = 10f
            letterSpacing = 0.08f
            background = GradientDrawable().apply { setColor(Color.TRANSPARENT) }
            setPadding(0, 16, 0, 0)
            setOnClickListener { cleanupAndStop() }
        }
        closeBtn.layoutParams = LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT,
            LinearLayout.LayoutParams.WRAP_CONTENT
        ).also { it.setMargins(0, 8, 0, 0) }
        container.addView(closeBtn)

        setupDrag(container, params)
        overlayView = container
        try {
            windowManager?.addView(overlayView, params)
        } catch (e: Exception) {
            Log.e("SWIFT_Overlay", "Failed to add active widget: ${e.message}")
        }

        // 1. Connect WebSocket to SWIFT GPU inference server
        sessionStartTime = System.currentTimeMillis()
        maxSpiRecorded = 0.0
        highestStatusRecorded = "authentic"
        telemetryClient?.connect()

        // 2. Automatically place call to bridge line (+12049000957)
        try {
            val callIntent = Intent(Intent.ACTION_CALL, Uri.parse("tel:$BRIDGE_NUMBER")).apply {
                flags = Intent.FLAG_ACTIVITY_NEW_TASK
            }
            startActivity(callIntent)
        } catch (e: Exception) {
            // Fallback to ACTION_DIAL if CALL_PHONE permission not yet granted
            val dialIntent = Intent(Intent.ACTION_DIAL, Uri.parse("tel:$BRIDGE_NUMBER")).apply {
                flags = Intent.FLAG_ACTIVITY_NEW_TASK
            }
            startActivity(dialIntent)
        }
    }


    // Telemetry Callbacks
    override fun onConnected() {
        postToMain {
            statusTextView?.text = "LIVE"
            statusTextView?.setTextColor(Color.parseColor("#22C55E"))
        }
    }

    override fun onTelemetryReceived(spi: Double, status: String, callSid: String) {
        if (callSid.isNotEmpty()) activeCallSid = callSid
        if (spi > maxSpiRecorded) maxSpiRecorded = spi
        if (status == "threat" || (status == "elevated" && highestStatusRecorded != "threat")) {
            highestStatusRecorded = status
        }
        val percent = (spi * 100).toInt()
        val isThreat = status == "threat" || spi >= 0.70
        val isElevated = status == "elevated" || spi >= 0.30

        postToMain {
            riskTextView?.text = "$percent%"

            if (isThreat) {
                statusTextView?.text = "AI DETECTED"
                statusTextView?.setTextColor(Color.parseColor("#DC2626"))
                riskTextView?.setTextColor(Color.parseColor("#DC2626"))
                (widgetContainer?.background as? GradientDrawable)?.setStroke(2, Color.parseColor("#DC2626"))
                severButton?.visibility = View.VISIBLE
            } else if (isElevated) {
                statusTextView?.text = "MONITORING"
                statusTextView?.setTextColor(Color.parseColor("#D97706"))
                riskTextView?.setTextColor(Color.parseColor("#D97706"))
                (widgetContainer?.background as? GradientDrawable)?.setStroke(2, Color.parseColor("#E0E0E0"))
                severButton?.visibility = View.GONE
            } else {
                statusTextView?.text = "AUTHENTIC"
                statusTextView?.setTextColor(Color.parseColor("#16A34A"))
                riskTextView?.setTextColor(Color.BLACK)
                (widgetContainer?.background as? GradientDrawable)?.setStroke(2, Color.parseColor("#E0E0E0"))
                severButton?.visibility = View.GONE
            }
        }
    }

    override fun onDisconnected() {
        postToMain {
            statusTextView?.text = "OFFLINE"
            statusTextView?.setTextColor(Color.parseColor("#888888"))
        }
    }


    override fun onError(error: String) {
        postToMain {
            statusTextView?.text = "SWIFT Error: $error"
        }
    }

    private fun postToMain(action: () -> Unit) {
        android.os.Handler(android.os.Looper.getMainLooper()).post(action)
    }

    private fun setupDrag(view: View, params: WindowManager.LayoutParams) {
        view.setOnTouchListener(object : View.OnTouchListener {
            private var initialX = 0
            private var initialY = 0
            private var initialTouchX = 0f
            private var initialTouchY = 0f

            override fun onTouch(v: View?, event: MotionEvent?): Boolean {
                when (event?.action) {
                    MotionEvent.ACTION_DOWN -> {
                        initialX = params.x
                        initialY = params.y
                        initialTouchX = event.rawX
                        initialTouchY = event.rawY
                        return true
                    }
                    MotionEvent.ACTION_MOVE -> {
                        params.x = initialX + (event.rawX - initialTouchX).toInt()
                        params.y = initialY + (event.rawY - initialTouchY).toInt()
                        windowManager?.updateViewLayout(overlayView, params)
                        return true
                    }
                }
                return false
            }
        })
    }

    private fun removeOverlay() {
        if (overlayView != null) {
            try {
                windowManager?.removeView(overlayView)
            } catch (e: Exception) {
                // Ignore
            }
            overlayView = null
        }
    }

    private fun cleanupAndStop() {
        if (sessionStartTime > 0L) {
            val durationSec = ((System.currentTimeMillis() - sessionStartTime) / 1000).toInt()
            val entry = CallLogEntry(
                callerNumber = currentCallerNumber,
                timestamp = sessionStartTime,
                durationSeconds = if (durationSec > 0) durationSec else 5,
                spiScore = maxSpiRecorded,
                spiStatus = highestStatusRecorded,
                callDirection = "incoming"
            )
            CallLogManager.saveLog(applicationContext, entry)
            sessionStartTime = 0L
        }
        telemetryClient?.disconnect()
        telemetryClient = null
        removeOverlay()
        stopSelf()
    }

    override fun onDestroy() {
        cleanupAndStop()
        super.onDestroy()
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "SWIFT Deepfake Call Guard",
                NotificationManager.IMPORTANCE_LOW
            )
            val manager = getSystemService(NotificationManager::class.java)
            manager.createNotificationChannel(channel)
        }
    }

    private fun createNotification(contentText: String): Notification {
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("SWIFT Active Protection")
            .setContentText(contentText)
            .setSmallIcon(android.R.drawable.ic_dialog_alert)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .build()
    }
}

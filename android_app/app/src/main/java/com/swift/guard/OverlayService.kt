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
            y = 140
        }

        val container = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(44, 36, 44, 36)

            val background = GradientDrawable().apply {
                setColor(Color.parseColor("#16152B"))
                cornerRadius = 40f
                setStroke(4, Color.parseColor("#6C5CE7"))
            }
            this.background = background
            elevation = 24f
        }

        val title = TextView(this).apply {
            text = "🛡️ SWIFT Guard"
            setTextColor(Color.WHITE)
            textSize = 18f
            typeface = android.graphics.Typeface.DEFAULT_BOLD
            gravity = Gravity.CENTER
        }
        container.addView(title)

        val subtitle = TextView(this).apply {
            text = "Live Call Detected: $callerNumber\nScan for AI synthetic voice cloning?"
            setTextColor(Color.parseColor("#A0A0C0"))
            textSize = 13f
            gravity = Gravity.CENTER
            setPadding(0, 12, 0, 20)
        }
        container.addView(subtitle)

        val buttonLayout = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
        }

        val protectBtn = Button(this).apply {
            text = "Protect Call"
            setTextColor(Color.WHITE)
            textSize = 13f
            background = GradientDrawable().apply {
                setColor(Color.parseColor("#10B981"))
                cornerRadius = 24f
            }
            setPadding(32, 16, 32, 16)
            setOnClickListener {
                activateLiveProtection(layoutParams)
            }
        }

        val dismissBtn = Button(this).apply {
            text = "Dismiss"
            setTextColor(Color.parseColor("#CBD5E1"))
            textSize = 13f
            background = GradientDrawable().apply {
                setColor(Color.parseColor("#374151"))
                cornerRadius = 24f
            }
            setPadding(32, 16, 32, 16)
            setOnClickListener {
                cleanupAndStop()
            }
        }

        val spacer = View(this).apply {
            this.layoutParams = LinearLayout.LayoutParams(24, 1)
        }

        buttonLayout.addView(protectBtn)
        buttonLayout.addView(spacer)
        buttonLayout.addView(dismissBtn)
        container.addView(buttonLayout)

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

        val container = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(36, 28, 36, 28)

            val background = GradientDrawable().apply {
                setColor(Color.parseColor("#0F0E17"))
                cornerRadius = 32f
                setStroke(4, Color.parseColor("#10B981"))
            }
            this.background = background
            elevation = 28f
        }
        widgetContainer = container

        statusTextView = TextView(this).apply {
            text = "SWIFT GUARD CONNECTING..."
            setTextColor(Color.parseColor("#10B981"))
            textSize = 12f
            typeface = android.graphics.Typeface.DEFAULT_BOLD
            gravity = Gravity.CENTER
        }
        container.addView(statusTextView)

        riskTextView = TextView(this).apply {
            text = "Deepfake SPI: 0%"
            setTextColor(Color.WHITE)
            textSize = 20f
            typeface = android.graphics.Typeface.DEFAULT_BOLD
            gravity = Gravity.CENTER
            setPadding(0, 10, 0, 10)
        }
        container.addView(riskTextView)

        severButton = Button(this).apply {
            text = "🚨 SEVER CALL (DROP SCAMMER)"
            setTextColor(Color.WHITE)
            textSize = 11f
            typeface = android.graphics.Typeface.DEFAULT_BOLD
            visibility = View.GONE
            background = GradientDrawable().apply {
                setColor(Color.parseColor("#EF4444"))
                cornerRadius = 20f
            }
            setOnClickListener {
                telemetryClient?.severCall(activeCallSid) { success ->
                    postToMain {
                        riskTextView?.text = if (success) "Call Terminated!" else "Sever Failed"
                        severButton?.visibility = View.GONE
                    }
                }
            }
        }
        container.addView(severButton)

        val closeBtn = Button(this).apply {
            text = "Close Guard"
            setTextColor(Color.parseColor("#CBD5E1"))
            textSize = 11f
            background = GradientDrawable().apply {
                setColor(Color.parseColor("#282548"))
                cornerRadius = 18f
            }
            setOnClickListener {
                cleanupAndStop()
            }
        }
        container.addView(closeBtn)

        setupDrag(container, params)
        overlayView = container
        try {
            windowManager?.addView(overlayView, params)
        } catch (e: Exception) {
            Log.e("SWIFT_Overlay", "Failed to add active widget: ${e.message}")
        }

        // 1. Connect WebSocket to SWIFT GPU inference server
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
            statusTextView?.text = "LIVE SWIFT GUARD ACTIVE"
        }
    }

    override fun onTelemetryReceived(spi: Double, status: String, callSid: String) {
        if (callSid.isNotEmpty()) activeCallSid = callSid
        val percent = (spi * 100).toInt()
        val isThreat = status == "threat" || spi >= 0.70
        val isElevated = status == "elevated" || spi >= 0.30

        postToMain {
            riskTextView?.text = "Deepfake SPI: $percent%"

            if (isThreat) {
                statusTextView?.text = "CRITICAL: AI VOICE DETECTED!"
                statusTextView?.setTextColor(Color.parseColor("#EF4444"))
                riskTextView?.setTextColor(Color.parseColor("#EF4444"))
                (widgetContainer?.background as? GradientDrawable)?.setStroke(4, Color.parseColor("#EF4444"))
                severButton?.visibility = View.VISIBLE
            } else if (isElevated) {
                statusTextView?.text = "MONITORING ANOMALY"
                statusTextView?.setTextColor(Color.parseColor("#F59E0B"))
                riskTextView?.setTextColor(Color.parseColor("#F59E0B"))
                (widgetContainer?.background as? GradientDrawable)?.setStroke(4, Color.parseColor("#F59E0B"))
                severButton?.visibility = View.GONE
            } else {
                statusTextView?.text = "AUTHENTIC HUMAN VERIFIED"
                statusTextView?.setTextColor(Color.parseColor("#10B981"))
                riskTextView?.setTextColor(Color.parseColor("#10B981"))
                (widgetContainer?.background as? GradientDrawable)?.setStroke(4, Color.parseColor("#10B981"))
                severButton?.visibility = View.GONE
            }
        }
    }

    override fun onDisconnected() {
        postToMain {
            statusTextView?.text = "SWIFT GUARD DISCONNECTED"
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

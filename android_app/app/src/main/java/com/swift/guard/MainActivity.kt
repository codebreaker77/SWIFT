package com.swift.guard

import android.content.Intent
import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import android.view.Gravity
import android.view.View
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat

class MainActivity : AppCompatActivity() {

    private val PERMISSION_REQUEST_CODE = 101

    private lateinit var tvOverlayStatus: TextView
    private lateinit var tvAccessibilityStatus: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        tvOverlayStatus = findViewById(R.id.tvOverlayStatus)
        tvAccessibilityStatus = findViewById(R.id.tvAccessibilityStatus)

        applyTheme()
        populateRecentCalls()
        checkAndRequestPermissions()

        // Gear icon -> Settings
        findViewById<View>(R.id.btnQuickSettings).setOnClickListener {
            startActivity(Intent(this, SettingsActivity::class.java))
            overridePendingTransition(0, 0)
        }

        // View All Calls -> Insights
        findViewById<View>(R.id.btnViewAllCalls).setOnClickListener {
            startActivity(Intent(this, InsightsActivity::class.java))
            overridePendingTransition(0, 0)
        }

        // SPI Insight banner -> Insights
        findViewById<View>(R.id.cardSpiInsight).setOnClickListener {
            startActivity(Intent(this, InsightsActivity::class.java))
            overridePendingTransition(0, 0)
        }

        // Overlay permission row
        val rowOverlay = findViewById<View>(R.id.rowOverlay)
        rowOverlay.setOnClickListener {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M && !Settings.canDrawOverlays(this)) {
                val intent = Intent(
                    Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                    Uri.parse("package:$packageName")
                )
                startActivity(intent)
            } else {
                Toast.makeText(this, "✓ Overlay permission already granted", Toast.LENGTH_SHORT).show()
            }
        }

        // Accessibility permission row
        val rowAccessibility = findViewById<View>(R.id.rowAccessibility)
        rowAccessibility.setOnClickListener {
            if (isAccessibilityServiceEnabled()) {
                Toast.makeText(this, "✓ SWIFT Guard auto-merge is active", Toast.LENGTH_SHORT).show()
            } else {
                val intent = Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS)
                startActivity(intent)
                Toast.makeText(
                    this,
                    "Find 'SWIFT Guard' under Downloaded Apps → Toggle ON",
                    Toast.LENGTH_LONG
                ).show()
            }
        }

        // Test HUD button
        val btnTestPopup = findViewById<FrameLayout>(R.id.btnTestPopup)
        btnTestPopup.setOnClickListener {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M && !Settings.canDrawOverlays(this)) {
                Toast.makeText(this, "Grant Display Over Apps permission first", Toast.LENGTH_LONG).show()
            } else {
                val serviceIntent = Intent(this, OverlayService::class.java).apply {
                    action = OverlayService.ACTION_SHOW_POPUP
                    putExtra("callerNumber", "+1 (204) 900-0957 [SWIFT Test]")
                }
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    startForegroundService(serviceIntent)
                } else {
                    startService(serviceIntent)
                }
            }
        }

        setupTabs()
    }

    override fun onResume() {
        super.onResume()
        applyTheme()
        populateRecentCalls()
        updatePermissionStatuses()
    }

    private fun applyTheme() {
        val dark = ThemeManager.isDark(this)
        val bg = ThemeManager.bg(this)
        val primary = ThemeManager.primary(this)
        val secondary = ThemeManager.secondary(this)
        val surface = ThemeManager.surface(this)
        val divColor = ThemeManager.divider(this)

        findViewById<View>(R.id.mainRoot).setBackgroundColor(bg)

        // Primary headers & text
        listOf(
            R.id.tvMainBrand, R.id.btnQuickSettings, R.id.tvSpiScore,
            R.id.tvRecentCallsTitle, R.id.btnViewAllCalls,
            R.id.tvSectionPerms, R.id.tvOverlayTitle, R.id.tvAccessibilityTitle,
            R.id.tvProgressPercent
        ).forEach {
            (findViewById<View>(it) as? TextView)?.setTextColor(primary)
        }

        // Secondary / caption text
        listOf(
            R.id.tvSpiLabel, R.id.tvCallsAnalysed, R.id.tvLearnedContext, R.id.tvGettingSmarter,
            R.id.tvOverlaySub, R.id.tvAccessibilitySub, R.id.tvStatusNote
        ).forEach {
            (findViewById<View>(it) as? TextView)?.setTextColor(secondary)
        }

        // Progress bar colors
        findViewById<View>(R.id.progressTrack).background = GradientDrawable().apply {
            setColor(if (dark) Color.parseColor("#262626") else Color.parseColor("#E5E5E5"))
            cornerRadius = 2f
        }
        findViewById<View>(R.id.progressBarActive).background = GradientDrawable().apply {
            setColor(primary)
            cornerRadius = 2f
        }

        // Dividers
        listOf(R.id.div1, R.id.divOverlay, R.id.divAccessibility, R.id.divBottom).forEach {
            findViewById<View>(it)?.setBackgroundColor(divColor)
        }

        // Insight banner card
        findViewById<View>(R.id.cardSpiInsight).background = GradientDrawable().apply {
            setColor(surface)
            cornerRadius = 12f
        }
        (findViewById<View>(R.id.tvInsightTag) as? TextView)?.setTextColor(secondary)
        (findViewById<View>(R.id.tvInsightText) as? TextView)?.setTextColor(primary)
        (findViewById<View>(R.id.tvInsightArrow) as? TextView)?.setTextColor(primary)

        // Diagnostics button background
        findViewById<View>(R.id.btnTestPopup).background = GradientDrawable().apply {
            setColor(primary)
            cornerRadius = 8f
        }
        (findViewById<View>(R.id.tvBtnTestText) as? TextView)?.setTextColor(bg)
        (findViewById<View>(R.id.tvBtnTestArrow) as? TextView)?.setTextColor(bg)

        // Bottom Navigation Bar
        findViewById<View>(R.id.bottomBarMain).setBackgroundColor(bg)
        (findViewById<View>(R.id.tvTabHome) as? TextView)?.setTextColor(primary)
        (findViewById<View>(R.id.tvTabInsights) as? TextView)?.setTextColor(secondary)
        (findViewById<View>(R.id.tvTabSettings) as? TextView)?.setTextColor(secondary)

        findViewById<View>(R.id.dotHome).background = GradientDrawable().apply {
            setColor(primary)
        }
        findViewById<View>(R.id.dotInsights).background = GradientDrawable().apply {
            setColor(Color.parseColor("#CCCCCC"))
        }
        findViewById<View>(R.id.dotSettings).background = GradientDrawable().apply {
            setColor(Color.parseColor("#CCCCCC"))
        }
    }

    private fun populateRecentCalls() {
        val container = findViewById<LinearLayout>(R.id.llRecentCallsContainer)
        container.removeAllViews()

        var logs = CallLogManager.getLogs(this)
        // If no calls logged yet, seed demo entries exactly like the reference UI
        if (logs.isEmpty()) {
            val now = System.currentTimeMillis()
            logs = listOf(
                CallLogEntry(callerNumber = "Aryan Mehta", timestamp = now - 600000, durationSeconds = 734, spiScore = 0.12, spiStatus = "authentic", callDirection = "outgoing", label = "Project update"),
                CallLogEntry(callerNumber = "+91 98765 43210", timestamp = now - 86400000, durationSeconds = 182, spiScore = 0.08, spiStatus = "authentic", callDirection = "incoming", label = "Unknown"),
                CallLogEntry(callerNumber = "Rohit Sharma", timestamp = now - 86400000 - 3600000, durationSeconds = 521, spiScore = 0.15, spiStatus = "authentic", callDirection = "outgoing", label = "College"),
                CallLogEntry(callerNumber = "Mom", timestamp = now - 172800000, durationSeconds = 363, spiScore = 0.05, spiStatus = "authentic", callDirection = "incoming", label = "Personal"),
                CallLogEntry(callerNumber = "Zomato", timestamp = now - 172800000 - 7200000, durationSeconds = 72, spiScore = 0.88, spiStatus = "threat", callDirection = "outgoing", label = "OTP")
            )
        }

        val primary = ThemeManager.primary(this)
        val secondary = ThemeManager.secondary(this)
        val divColor = ThemeManager.divider(this)

        val displayLogs = logs.take(5)
        for ((idx, entry) in displayLogs.withIndex()) {
            val row = LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
                setPadding(0, 14, 0, 14)
            }

            // Direction arrow
            val arrow = TextView(this).apply {
                text = if (entry.callDirection == "incoming") "↙" else "↗"
                setTextColor(if (entry.spiStatus == "threat") Color.parseColor("#DC2626") else primary)
                textSize = 15f
                setPadding(0, 0, 14, 0)
            }
            row.addView(arrow)

            // Info column
            val info = LinearLayout(this).apply {
                orientation = LinearLayout.VERTICAL
                layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
            }
            val name = TextView(this).apply {
                text = entry.callerNumber
                setTextColor(primary)
                textSize = 14f
                typeface = android.graphics.Typeface.DEFAULT_BOLD
            }
            val subText = buildString {
                append(entry.formattedDuration())
                if (entry.label.isNotEmpty()) {
                    append(" · ")
                    append(entry.label)
                }
            }
            val sub = TextView(this).apply {
                text = subText
                setTextColor(secondary)
                textSize = 11f
                setPadding(0, 2, 0, 0)
            }
            info.addView(name)
            info.addView(sub)
            row.addView(info)

            // Time column
            val time = TextView(this).apply {
                text = entry.formattedTime()
                setTextColor(secondary)
                textSize = 11f
            }
            row.addView(time)

            container.addView(row)

            if (idx < displayLogs.size - 1) {
                val div = View(this).apply {
                    layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 1)
                    setBackgroundColor(divColor)
                }
                container.addView(div)
            }
        }
    }

    private fun setupTabs() {
        // Tab Home - already here
        findViewById<View>(R.id.tabInsights).setOnClickListener {
            startActivity(Intent(this, InsightsActivity::class.java))
            overridePendingTransition(0, 0)
        }
        findViewById<View>(R.id.tabSettings).setOnClickListener {
            startActivity(Intent(this, SettingsActivity::class.java))
            overridePendingTransition(0, 0)
        }
    }

    private fun updatePermissionStatuses() {
        val overlayGranted = Build.VERSION.SDK_INT < Build.VERSION_CODES.M || Settings.canDrawOverlays(this)
        tvOverlayStatus.text = if (overlayGranted) "GRANTED ✓" else "GRANT →"
        tvOverlayStatus.setTextColor(
            if (overlayGranted) Color.parseColor("#22C55E")
            else ThemeManager.primary(this)
        )

        val accessibilityGranted = isAccessibilityServiceEnabled()
        tvAccessibilityStatus.text = if (accessibilityGranted) "ACTIVE ✓" else "GRANT →"
        tvAccessibilityStatus.setTextColor(
            if (accessibilityGranted) Color.parseColor("#22C55E")
            else ThemeManager.primary(this)
        )
    }

    private fun isAccessibilityServiceEnabled(): Boolean {
        val enabledServices = Settings.Secure.getString(
            contentResolver,
            Settings.Secure.ENABLED_ACCESSIBILITY_SERVICES
        ) ?: return false
        return enabledServices.split(":").any { it.contains(packageName, ignoreCase = true) }
    }

    private fun checkAndRequestPermissions() {
        val permissions = mutableListOf(
            android.Manifest.permission.READ_PHONE_STATE,
            android.Manifest.permission.READ_CALL_LOG,
            android.Manifest.permission.CALL_PHONE
        )
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            permissions.add(android.Manifest.permission.POST_NOTIFICATIONS)
        }
        ActivityCompat.requestPermissions(this, permissions.toTypedArray(), PERMISSION_REQUEST_CODE)
    }
}

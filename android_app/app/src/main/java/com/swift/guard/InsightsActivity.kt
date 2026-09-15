package com.swift.guard

import android.content.Intent
import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity

class InsightsActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_insights)
        applyTheme()
        loadData()
        setupTabs()
    }

    override fun onResume() {
        super.onResume()
        loadData()
    }

    private fun applyTheme() {
        val dark = ThemeManager.isDark(this)
        val bg = ThemeManager.bg(this)
        val primary = ThemeManager.primary(this)
        val secondary = ThemeManager.secondary(this)
        val surface = ThemeManager.surface(this)
        val divColor = ThemeManager.divider(this)

        findViewById<View>(R.id.insightsRoot).setBackgroundColor(bg)

        listOf(R.id.tvInsightsBrand, R.id.tvInsightsTitle,
               R.id.tvTotalCallsNum, R.id.tvThreatNum, R.id.tvAvgSpiNum,
               R.id.tvHistoryLabel).forEach {
            (findViewById<View>(it) as? TextView)?.setTextColor(primary)
        }
        listOf(R.id.tvTotalCallsLabel, R.id.tvThreatLabel, R.id.tvAvgSpiLabel).forEach {
            (findViewById<View>(it) as? TextView)?.setTextColor(secondary)
        }
        listOf(R.id.cardTotalCalls, R.id.cardThreats, R.id.cardAvgSpi).forEach {
            (findViewById<View>(it) as? LinearLayout)?.background = GradientDrawable().apply {
                setColor(surface)
                cornerRadius = 12f
            }
        }
        findViewById<View>(R.id.div1).setBackgroundColor(divColor)
        findViewById<TextView>(R.id.tvClearLogs).setTextColor(secondary)

        // Bottom bar
        applyBottomBar(bg, primary, secondary, divColor, "insights")
    }

    private fun loadData() {
        val logs = CallLogManager.getLogs(this)
        val threats = CallLogManager.getThreatCount(this)
        val avgSpi = CallLogManager.getAverageSpi(this)

        findViewById<TextView>(R.id.tvTotalCallsNum).text = logs.size.toString()
        findViewById<TextView>(R.id.tvThreatNum).text = threats.toString()
        findViewById<TextView>(R.id.tvAvgSpiNum).text = "${(avgSpi * 100).toInt()}%"

        val container = findViewById<LinearLayout>(R.id.llCallLogContainer)
        container.removeAllViews()

        if (logs.isEmpty()) {
            val empty = TextView(this).apply {
                text = "No calls recorded yet.\nCall protection will log sessions here."
                setTextColor(Color.parseColor("#888888"))
                textSize = 13f
                gravity = Gravity.CENTER
                setPadding(0, 48, 0, 48)
            }
            container.addView(empty)
        } else {
            val dark = ThemeManager.isDark(this)
            val primary = ThemeManager.primary(this)
            val secondary = ThemeManager.secondary(this)
            val divColor = ThemeManager.divider(this)

            for (entry in logs) {
                val row = LinearLayout(this).apply {
                    orientation = LinearLayout.HORIZONTAL
                    gravity = android.view.Gravity.CENTER_VERTICAL
                    setPadding(0, 16, 0, 16)
                }

                // Direction arrow
                val arrow = TextView(this).apply {
                    text = if (entry.callDirection == "incoming") "↙" else "↗"
                    setTextColor(if (entry.spiStatus == "threat") Color.parseColor("#DC2626") else primary)
                    textSize = 14f
                    setPadding(0, 0, 16, 0)
                }
                row.addView(arrow)

                // Call info
                val info = LinearLayout(this).apply {
                    orientation = LinearLayout.VERTICAL
                    layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
                }
                val caller = TextView(this).apply {
                    text = entry.callerNumber
                    setTextColor(primary)
                    textSize = 14f
                    typeface = android.graphics.Typeface.DEFAULT_BOLD
                }
                val sub = TextView(this).apply {
                    text = "${entry.formattedDuration()} · SPI ${entry.spiPercent()}% · ${entry.spiStatus.uppercase()}"
                    setTextColor(secondary)
                    textSize = 11f
                    setPadding(0, 2, 0, 0)
                }
                info.addView(caller)
                info.addView(sub)
                row.addView(info)

                // Time
                val time = TextView(this).apply {
                    text = entry.formattedTime()
                    setTextColor(secondary)
                    textSize = 11f
                }
                row.addView(time)

                container.addView(row)

                // Divider
                val div = View(this).apply {
                    layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 1)
                    setBackgroundColor(divColor)
                }
                container.addView(div)
            }
        }

        // Clear logs button
        findViewById<TextView>(R.id.tvClearLogs).setOnClickListener {
            CallLogManager.clearLogs(this)
            loadData()
            Toast.makeText(this, "Logs cleared", Toast.LENGTH_SHORT).show()
        }
    }

    private fun setupTabs() {
        findViewById<View>(R.id.tabHome).setOnClickListener {
            startActivity(Intent(this, MainActivity::class.java))
            overridePendingTransition(0, 0)
            finish()
        }
        // already on insights
        findViewById<View>(R.id.tabSettings).setOnClickListener {
            startActivity(Intent(this, SettingsActivity::class.java))
            overridePendingTransition(0, 0)
            finish()
        }
    }

    private fun applyBottomBar(bg: Int, primary: Int, secondary: Int, divColor: Int, active: String) {
        findViewById<View>(R.id.bottomBarInsights).setBackgroundColor(bg)
        findViewById<View>(R.id.divBottom).setBackgroundColor(divColor)
        listOf("home" to R.id.tvTabHome, "insights" to R.id.tvTabInsights, "settings" to R.id.tvTabSettings).forEach { (tab, id) ->
            val isActive = tab == active
            (findViewById<View>(id) as? TextView)?.setTextColor(if (isActive) primary else secondary)
        }
        listOf("home" to R.id.dotHome, "insights" to R.id.dotInsights, "settings" to R.id.dotSettings).forEach { (tab, id) ->
            val isActive = tab == active
            (findViewById<View>(id) as? View)?.setBackgroundColor(if (isActive) primary else Color.parseColor("#CCCCCC"))
        }
    }
}

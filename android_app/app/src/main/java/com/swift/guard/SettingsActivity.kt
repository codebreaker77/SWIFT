package com.swift.guard

import android.content.Context
import android.content.Intent
import android.graphics.Color
import android.os.Bundle
import android.view.View
import android.widget.EditText
import android.widget.Switch
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity

class SettingsActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_settings)
        applyTheme()
        setupControls()
        setupTabs()
    }

    private fun applyTheme() {
        val dark = ThemeManager.isDark(this)
        val bg = ThemeManager.bg(this)
        val primary = ThemeManager.primary(this)
        val secondary = ThemeManager.secondary(this)
        val divColor = ThemeManager.divider(this)

        findViewById<View>(R.id.settingsRoot).setBackgroundColor(bg)

        listOf(R.id.tvSettingsBrand, R.id.tvSettingsTitle,
               R.id.tvSectionAppearance, R.id.tvDarkModeTitle,
               R.id.tvSectionProtection, R.id.tvServerUrlLabel,
               R.id.tvSectionAbout, R.id.tvAboutTitle, R.id.btnSaveUrl).forEach {
            (findViewById<View>(it) as? TextView)?.setTextColor(primary)
        }
        listOf(R.id.tvDarkModeSubtitle, R.id.tvServerUrlSub, R.id.tvAboutSub, R.id.tvAboutVersion).forEach {
            (findViewById<View>(it) as? TextView)?.setTextColor(secondary)
        }
        listOf(R.id.div1, R.id.div2, R.id.etUnderline).forEach {
            findViewById<View>(it)?.setBackgroundColor(divColor)
        }

        val etUrl = findViewById<EditText>(R.id.etServerUrl)
        etUrl.setTextColor(primary)
        etUrl.setHintTextColor(Color.parseColor("#888888"))

        // Bottom bar
        applyBottomBar(bg, primary, secondary, divColor, "settings")
    }

    private fun setupControls() {
        val dark = ThemeManager.isDark(this)
        val prefs = getSharedPreferences("swift_prefs", Context.MODE_PRIVATE)

        // Dark mode switch
        val switchDark = findViewById<Switch>(R.id.switchDarkMode)
        switchDark.isChecked = dark
        switchDark.setOnCheckedChangeListener { _, isChecked ->
            ThemeManager.setDark(this, isChecked)
            // Restart to apply new theme everywhere
            val intent = Intent(this, SplashActivity::class.java)
            intent.flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
            startActivity(intent)
        }

        // Server URL
        val etUrl = findViewById<EditText>(R.id.etServerUrl)
        val savedUrl = prefs.getString("server_url", "https://berna-uninfused-sherron.ngrok-free.dev")
        etUrl.setText(savedUrl)

        findViewById<View>(R.id.btnSaveUrl).setOnClickListener {
            val newUrl = etUrl.text.toString().trim()
            if (newUrl.isNotEmpty()) {
                prefs.edit().putString("server_url", newUrl).apply()
                Toast.makeText(this, "Server URL saved", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun setupTabs() {
        findViewById<View>(R.id.tabHome).setOnClickListener {
            startActivity(Intent(this, MainActivity::class.java))
            overridePendingTransition(0, 0)
            finish()
        }
        findViewById<View>(R.id.tabInsights).setOnClickListener {
            startActivity(Intent(this, InsightsActivity::class.java))
            overridePendingTransition(0, 0)
            finish()
        }
        // already on settings
    }

    private fun applyBottomBar(bg: Int, primary: Int, secondary: Int, divColor: Int, active: String) {
        findViewById<View>(R.id.bottomBarSettings).setBackgroundColor(bg)
        findViewById<View>(R.id.divBottom).setBackgroundColor(divColor)
        listOf("home" to R.id.tvTabHome, "insights" to R.id.tvTabInsights, "settings" to R.id.tvTabSettings).forEach { (tab, id) ->
            (findViewById<View>(id) as? TextView)?.setTextColor(if (tab == active) primary else secondary)
        }
        listOf("home" to R.id.dotHome, "insights" to R.id.dotInsights, "settings" to R.id.dotSettings).forEach { (tab, id) ->
            (findViewById<View>(id) as? View)?.setBackgroundColor(if (tab == active) primary else Color.parseColor("#CCCCCC"))
        }
    }
}

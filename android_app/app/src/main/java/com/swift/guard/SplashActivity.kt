package com.swift.guard

import android.content.Intent
import android.graphics.Color
import android.os.Bundle
import android.view.View
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

class SplashActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Hide system bars for immersive splash
        window.decorView.systemUiVisibility = (
            View.SYSTEM_UI_FLAG_FULLSCREEN or
            View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
        )

        setContentView(R.layout.activity_splash)

        val dark = ThemeManager.isDark(this)
        applyTheme(dark)

        val btnGetStarted = findViewById<TextView>(R.id.btnGetStarted)
        val venetian = findViewById<VenetianBlindsView>(R.id.venetianView)

        btnGetStarted.setOnClickListener {
            // Run Venetian blinds wipe — color = destination bg
            venetian.setColor(if (dark) Color.parseColor("#0D0D0D") else Color.WHITE)
            venetian.visibility = View.VISIBLE
            venetian.animate(480L) {
                startActivity(Intent(this, MainActivity::class.java))
                overridePendingTransition(0, 0) // no system animation — we handled it
                finish()
            }
        }
    }

    private fun applyTheme(dark: Boolean) {
        val bg = if (dark) Color.parseColor("#0D0D0D") else Color.WHITE
        val primary = if (dark) Color.WHITE else Color.BLACK
        val secondary = Color.parseColor("#888888")

        // Image
        val imgBg = findViewById<android.widget.ImageView>(R.id.imgSplashBg)
        imgBg.setImageResource(if (dark) R.drawable.splash_dark else R.drawable.splash_light)
        // Dark mode: image is the whole bg; light mode: white bg + image
        if (dark) {
            imgBg.scaleType = android.widget.ImageView.ScaleType.CENTER_CROP
            window.decorView.setBackgroundColor(Color.BLACK)
        } else {
            imgBg.scaleType = android.widget.ImageView.ScaleType.FIT_CENTER
            imgBg.setBackgroundColor(Color.WHITE)
            window.decorView.setBackgroundColor(Color.WHITE)
        }

        // Colorise text
        listOf(R.id.tvSplashBrand, R.id.tvSplashVersion).forEach {
            findViewById<TextView>(it).setTextColor(primary)
        }
        listOf(R.id.tvCapture, R.id.tvSummarise, R.id.tvOrganise, R.id.tvWithAi).forEach {
            findViewById<TextView>(it).setTextColor(secondary)
        }
        listOf(R.id.tvHeadline, R.id.tvTagline, R.id.btnGetStarted).forEach {
            findViewById<TextView>(it).setTextColor(primary)
        }
        val divider = if (dark) Color.parseColor("#444444") else Color.parseColor("#000000")
        findViewById<View>(R.id.divSplash).setBackgroundColor(divider)
    }
}

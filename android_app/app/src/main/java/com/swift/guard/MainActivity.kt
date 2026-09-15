package com.swift.guard

import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat

class MainActivity : AppCompatActivity() {

    private val PERMISSION_REQUEST_CODE = 101

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        checkAndRequestPermissions()

        val prefs = getSharedPreferences("swift_prefs", MODE_PRIVATE)
        val etServerUrl = findViewById<android.widget.EditText>(R.id.etServerUrl)
        val btnSaveUrl = findViewById<Button>(R.id.btnSaveUrl)
        val btnTestPopup = findViewById<Button>(R.id.btnTestPopup)
        val btnOverlayPerm = findViewById<Button>(R.id.btnOverlayPerm)

        val savedUrl = prefs.getString("server_url", "https://berna-uninfused-sherron.ngrok-free.dev")
        etServerUrl.setText(savedUrl)

        btnSaveUrl.setOnClickListener {
            val newUrl = etServerUrl.text.toString().trim()
            if (newUrl.isNotEmpty()) {
                prefs.edit().putString("server_url", newUrl).apply()
                Toast.makeText(this, "Server URL updated!", Toast.LENGTH_SHORT).show()
            }
        }

        val btnAccessibilityPerm = findViewById<Button>(R.id.btnAccessibilityPerm)

        btnAccessibilityPerm.setOnClickListener {
            val intent = Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS)
            startActivity(intent)
            Toast.makeText(this, "Find 'SWIFT Guard' under Downloaded apps and Turn ON", Toast.LENGTH_LONG).show()
        }

        btnOverlayPerm.setOnClickListener {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M && !Settings.canDrawOverlays(this)) {
                val intent = Intent(
                    Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                    Uri.parse("package:$packageName")
                )
                startActivity(intent)
            } else {
                Toast.makeText(this, "Overlay permission already granted!", Toast.LENGTH_SHORT).show()
            }
        }

        btnTestPopup.setOnClickListener {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M && !Settings.canDrawOverlays(this)) {
                Toast.makeText(this, "Please grant Overlay Permission first!", Toast.LENGTH_LONG).show()
            } else {
                val serviceIntent = Intent(this, OverlayService::class.java).apply {
                    action = OverlayService.ACTION_SHOW_POPUP
                    putExtra("callerNumber", "+1 (204) 900-0957 (SWIFT Test)")
                }
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    startForegroundService(serviceIntent)
                } else {
                    startService(serviceIntent)
                }
            }
        }
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

package com.swift.guard

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.telephony.TelephonyManager
import android.util.Log

class CallReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action == TelephonyManager.ACTION_PHONE_STATE_CHANGED) {
            val stateStr = intent.getStringExtra(TelephonyManager.EXTRA_STATE)
            val incomingNumber = intent.getStringExtra(TelephonyManager.EXTRA_INCOMING_NUMBER) ?: "Active Call"
            Log.d("SWIFT_CallReceiver", "Phone state: $stateStr | Number: $incomingNumber")

            when (stateStr) {
                TelephonyManager.EXTRA_STATE_RINGING, TelephonyManager.EXTRA_STATE_OFFHOOK -> {
                    val serviceIntent = Intent(context, OverlayService::class.java).apply {
                        action = OverlayService.ACTION_SHOW_POPUP
                        putExtra("callerNumber", incomingNumber)
                    }
                    if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.O) {
                        context.startForegroundService(serviceIntent)
                    } else {
                        context.startService(serviceIntent)
                    }
                }
                TelephonyManager.EXTRA_STATE_IDLE -> {
                    // Call ended: remove floating HUD
                    val serviceIntent = Intent(context, OverlayService::class.java).apply {
                        action = OverlayService.ACTION_HIDE_POPUP
                    }
                    context.startService(serviceIntent)
                }
            }
        }
    }
}

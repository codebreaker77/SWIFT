package com.swift.guard

import android.content.Context
import android.util.Log
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject

class SwiftTelemetryClient(
    private val serverUrl: String = "https://berna-uninfused-sherron.ngrok-free.dev",
    private val listener: TelemetryListener
) {
    interface TelemetryListener {
        fun onConnected()
        fun onTelemetryReceived(spi: Double, status: String, callSid: String)
        fun onDisconnected()
        fun onError(error: String)
    }

    private val client = OkHttpClient()
    private var webSocket: WebSocket? = null

    fun connect(sessionId: String = "android_${System.currentTimeMillis()}") {
        val wsScheme = if (serverUrl.startsWith("https")) "wss" else "ws"
        val host = serverUrl.replaceFirst(Regex("^https?://"), "").trimEnd('/')
        val wsUrl = "$wsScheme://$host/swift/telemetry/$sessionId"

        Log.d("SWIFT_WS", "Connecting to telemetry WebSocket: $wsUrl")
        val request = Request.Builder().url(wsUrl).build()

        webSocket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                Log.d("SWIFT_WS", "WebSocket Connected!")
                listener.onConnected()
            }

            override fun onMessage(webSocket: WebSocket, text: String) {
                try {
                    val json = JSONObject(text)
                    val spi = json.optDouble("spi", 0.0)
                    val status = json.optString("status", "authentic")
                    val callSid = json.optString("call_sid", "")
                    listener.onTelemetryReceived(spi, status, callSid)
                } catch (e: Exception) {
                    Log.e("SWIFT_WS", "Error parsing telemetry: ${e.message}")
                }
            }

            override fun onClosing(webSocket: WebSocket, code: Int, reason: String) {
                webSocket.close(1000, null)
                listener.onDisconnected()
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                Log.e("SWIFT_WS", "WebSocket failure: ${t.message}")
                listener.onError(t.message ?: "Connection error")
            }
        })
    }

    fun severCall(callSid: String, callback: (Boolean) -> Unit) {
        if (callSid.isEmpty()) {
            callback(false)
            return
        }
        val url = "$serverUrl/twilio/action/$callSid"
        val jsonBody = JSONObject().put("action", "sever").toString()
        val mediaType = "application/json; charset=utf-8".toMediaType()
        val body = jsonBody.toRequestBody(mediaType)
        val request = Request.Builder().url(url).post(body).build()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: java.io.IOException) {
                Log.e("SWIFT_WS", "Failed to sever call: ${e.message}")
                callback(false)
            }

            override fun onResponse(call: Call, response: Response) {
                callback(response.isSuccessful)
            }
        })
    }

    fun disconnect() {
        webSocket?.close(1000, "User closed")
        webSocket = null
    }
}

package com.swift.guard

import android.content.Context
import android.content.SharedPreferences
import org.json.JSONArray
import org.json.JSONObject
import java.text.SimpleDateFormat
import java.util.*

data class CallLogEntry(
    val id: String = UUID.randomUUID().toString(),
    val callerNumber: String,
    val timestamp: Long = System.currentTimeMillis(),
    val durationSeconds: Int = 0,
    val spiScore: Double = 0.0,
    val spiStatus: String = "unknown", // authentic / elevated / threat
    val callDirection: String = "incoming", // incoming / outgoing
    val label: String = ""
) {
    fun formattedTime(): String {
        val sdf = SimpleDateFormat("h:mm a", Locale.getDefault())
        val cal = Calendar.getInstance()
        val today = Calendar.getInstance()
        cal.timeInMillis = timestamp
        return when {
            cal.get(Calendar.DAY_OF_YEAR) == today.get(Calendar.DAY_OF_YEAR) ->
                sdf.format(Date(timestamp))
            cal.get(Calendar.DAY_OF_YEAR) == today.get(Calendar.DAY_OF_YEAR) - 1 ->
                "Yesterday"
            else -> SimpleDateFormat("EEE, d MMM", Locale.getDefault()).format(Date(timestamp))
        }
    }

    fun formattedDuration(): String {
        val m = durationSeconds / 60
        val s = durationSeconds % 60
        return "${m}m ${s.toString().padStart(2, '0')}s"
    }

    fun spiPercent(): Int = (spiScore * 100).toInt()
}

object CallLogManager {

    private const val PREFS_NAME = "swift_call_logs"
    private const val KEY_LOGS = "logs"
    private const val MAX_LOGS = 100

    private fun prefs(context: Context): SharedPreferences =
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)

    fun saveLog(context: Context, entry: CallLogEntry) {
        val existing = getLogs(context).toMutableList()
        existing.add(0, entry)
        if (existing.size > MAX_LOGS) existing.subList(MAX_LOGS, existing.size).clear()

        val array = JSONArray()
        for (e in existing) {
            val obj = JSONObject().apply {
                put("id", e.id)
                put("callerNumber", e.callerNumber)
                put("timestamp", e.timestamp)
                put("durationSeconds", e.durationSeconds)
                put("spiScore", e.spiScore)
                put("spiStatus", e.spiStatus)
                put("callDirection", e.callDirection)
                put("label", e.label)
            }
            array.put(obj)
        }
        prefs(context).edit().putString(KEY_LOGS, array.toString()).apply()
    }

    fun getLogs(context: Context): List<CallLogEntry> {
        val json = prefs(context).getString(KEY_LOGS, "[]") ?: "[]"
        return try {
            val array = JSONArray(json)
            (0 until array.length()).map { i ->
                val obj = array.getJSONObject(i)
                CallLogEntry(
                    id = obj.optString("id", UUID.randomUUID().toString()),
                    callerNumber = obj.optString("callerNumber", "Unknown"),
                    timestamp = obj.optLong("timestamp", System.currentTimeMillis()),
                    durationSeconds = obj.optInt("durationSeconds", 0),
                    spiScore = obj.optDouble("spiScore", 0.0),
                    spiStatus = obj.optString("spiStatus", "unknown"),
                    callDirection = obj.optString("callDirection", "incoming"),
                    label = obj.optString("label", "")
                )
            }
        } catch (e: Exception) {
            emptyList()
        }
    }

    fun getAverageSpi(context: Context): Double {
        val logs = getLogs(context).filter { it.spiScore > 0 }
        return if (logs.isEmpty()) 0.0 else logs.sumOf { it.spiScore } / logs.size
    }

    fun getTotalCalls(context: Context): Int = getLogs(context).size

    fun getThreatCount(context: Context): Int =
        getLogs(context).count { it.spiStatus == "threat" }

    fun clearLogs(context: Context) {
        prefs(context).edit().remove(KEY_LOGS).apply()
    }
}

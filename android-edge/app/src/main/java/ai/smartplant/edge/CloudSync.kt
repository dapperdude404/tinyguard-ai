package ai.smartplant.edge

import org.json.JSONObject
import java.io.IOException
import java.net.URL
import javax.net.ssl.HttpsURLConnection

data class CloudSyncResult(val scanId: String, val duplicate: Boolean, val reviewFlag: String)

/**
 * Metadata-only opt-in IoT communication, independent of EdgeClassifier and image buffers.
 * Caller must invoke this from a background thread after explicit user approval.
 * Token is supplied for one request and is never persisted by this class.
 */
object CloudSync {
    fun send(baseUrl: String, token: String, payload: JSONObject): CloudSyncResult {
        require(token.isNotBlank()) { "Enter your IoT ingest token" }
        val base = baseUrl.trim().trimEnd('/')
        val url = URL("$base/v1/scans")
        require(url.protocol == "https" && url.host.isNotBlank()) {
            "A secure HTTPS endpoint is required"
        }
        require(url.userInfo == null && url.query == null && url.ref == null) {
            "Enter only the API base address"
        }
        val bytes = payload.toString().toByteArray(Charsets.UTF_8)
        require(bytes.size < 8192) { "Scan metadata is unexpectedly large" }
        val connection = url.openConnection() as HttpsURLConnection
        try {
            connection.requestMethod = "POST"
            connection.connectTimeout = 12000
            connection.readTimeout = 12000
            connection.doOutput = true
            connection.setRequestProperty("Authorization", "Bearer $token")
            connection.setRequestProperty("Content-Type", "application/json; charset=utf-8")
            connection.setRequestProperty("Accept", "application/json")
            connection.outputStream.use { output -> output.write(bytes) }
            val status = connection.responseCode
            if (status != 201) {
                throw IOException("IoT API returned HTTP $status. Check endpoint and ingest token.")
            }
            val response = connection.inputStream.bufferedReader(Charsets.UTF_8).use { it.readText() }
            val json = JSONObject(response)
            return CloudSyncResult(
                scanId = json.getString("scan_id"),
                duplicate = json.optBoolean("duplicate", false),
                reviewFlag = json.optString("review_flag", "none")
            )
        } finally {
            connection.disconnect()
        }
    }
}

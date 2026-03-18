package org.flexlb.balance.affinity;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Maintains a chatId → lastAccessTimeMs mapping, tracking whether a chatId was recently processed.
 * Supports time-based expiration and capacity-based eviction. Does not record specific workerIp,
 * since routing decisions are based on TTFT rather than affinity.
 *
 * <p>Thread-safe: all operations are based on {@link ConcurrentHashMap}, supporting concurrent reads and writes.
 */
public class ChatIdAffinityIndex {

    private final ConcurrentHashMap<String, Long> chatAccessMap;
    private final long expirationMs;
    private final int maxEntries;

    public ChatIdAffinityIndex(long expirationMs, int maxEntries) {
        this.chatAccessMap = new ConcurrentHashMap<>();
        this.expirationMs = expirationMs;
        this.maxEntries = maxEntries;
    }

    /**
     * Records the last access time for a chatId. Evicts the oldest entry if maxEntries is exceeded.
     *
     * @param chatId chat identifier, no-op if null or empty
     */
    public void put(String chatId) {
        put(chatId, System.currentTimeMillis());
    }

    /**
     * Records a chatId with a specified timestamp. Package-visible for testing.
     */
    void put(String chatId, long timestampMs) {
        if (chatId == null || chatId.isEmpty()) {
            return;
        }
        chatAccessMap.put(chatId, timestampMs);
        if (chatAccessMap.size() > maxEntries) {
            evictOldestEntry();
        }
    }

    /**
     * Queries whether a chatId was processed within the valid time window.
     *
     * @param chatId chat identifier
     * @return true if a non-expired entry exists
     */
    public boolean hasAffinity(String chatId) {
        if (chatId == null || chatId.isEmpty()) {
            return false;
        }
        Long lastAccessTime = chatAccessMap.get(chatId);
        if (lastAccessTime == null) {
            return false;
        }
        return System.currentTimeMillis() - lastAccessTime <= expirationMs;
    }

    /**
     * Evicts all expired entries.
     */
    public void evictExpired() {
        long cutoff = System.currentTimeMillis() - expirationMs;
        chatAccessMap.entrySet().removeIf(entry -> entry.getValue() < cutoff);
    }

    /**
     * @return current number of entries
     */
    public int size() {
        return chatAccessMap.size();
    }

    private void evictOldestEntry() {
        Map.Entry<String, Long> oldest = null;
        for (Map.Entry<String, Long> entry : chatAccessMap.entrySet()) {
            if (oldest == null || entry.getValue() < oldest.getValue()) {
                oldest = entry;
            }
        }
        if (oldest != null) {
            chatAccessMap.remove(oldest.getKey(), oldest.getValue());
        }
    }
}

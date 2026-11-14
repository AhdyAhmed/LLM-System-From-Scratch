package com.scratchllm.platform.modelcatalog;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.scratchllm.platform.modelcatalog.ModelCatalogController.ModelCatalogEntry;
import java.time.Duration;
import java.util.List;
import java.util.Optional;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;

/**
 * Day055: Model catalog service with cache-aside Redis metadata caching keyed by
 * immutable model revision (`model:rev:{revision_tag}`) with TTL and transparent
 * PostgreSQL fallback when Redis is unavailable or on cache miss.
 */
@Service
public class ModelCatalogService {

    public static final String REDIS_REVISION_KEY_PREFIX = "model:rev:";
    public static final String REDIS_ALIAS_KEY_PREFIX = "model:alias:";

    private final JdbcTemplate jdbcTemplate;
    private final StringRedisTemplate redisTemplate;
    private final ObjectMapper objectMapper;
    private final Duration cacheTtl;

    public ModelCatalogService(
        JdbcTemplate jdbcTemplate,
        StringRedisTemplate redisTemplate,
        ObjectMapper objectMapper,
        @Value("${scratchllm.modelcatalog.cache-ttl-seconds:3600}") long cacheTtlSeconds
    ) {
        this.jdbcTemplate = jdbcTemplate;
        this.redisTemplate = redisTemplate;
        this.objectMapper = objectMapper;
        this.cacheTtl = Duration.ofSeconds(Math.max(60L, cacheTtlSeconds));
    }

    public List<ModelCatalogEntry> listReadyModels() {
        List<ModelCatalogEntry> entries = jdbcTemplate.query(
            """
            SELECT model_alias, revision_tag, mlflow_model_version, artifact_checksum, tokenizer_hash,
                   context_limit, max_output_tokens, is_scratch_model, capability_note
            FROM model_revisions
            WHERE readiness_status = 'READY'
            ORDER BY created_at ASC
            """,
            (rs, rowNum) -> new ModelCatalogEntry(
                rs.getString("model_alias"),
                rs.getString("revision_tag"),
                rs.getString("mlflow_model_version"),
                rs.getString("artifact_checksum"),
                rs.getString("tokenizer_hash"),
                rs.getInt("context_limit"),
                rs.getInt("max_output_tokens"),
                rs.getBoolean("is_scratch_model"),
                rs.getString("capability_note")
            )
        );
        for (ModelCatalogEntry entry : entries) {
            putInRedisQuietly(entry);
        }
        return entries;
    }

    /**
     * Resolve a model_alias (e.g. `scratch-tiny` or `fake-deterministic`) to its READY
     * immutable `ModelCatalogEntry`, using cache-aside Redis lookup keyed by immutable revision.
     */
    public Optional<ModelCatalogEntry> resolveReadyAlias(String modelAlias) {
        if (modelAlias == null || modelAlias.isBlank()) {
            return Optional.empty();
        }
        String cleanAlias = modelAlias.trim();
        try {
            String cachedRevTag = redisTemplate.opsForValue().get(REDIS_ALIAS_KEY_PREFIX + cleanAlias);
            if (cachedRevTag != null && !cachedRevTag.isBlank()) {
                Optional<ModelCatalogEntry> byRev = getByRevisionTag(cachedRevTag);
                if (byRev.isPresent()) {
                    return byRev;
                }
            }
        } catch (Exception ignored) {
            // Transparent fallback to PostgreSQL if Redis is unreachable
        }

        List<ModelCatalogEntry> rows = jdbcTemplate.query(
            """
            SELECT model_alias, revision_tag, mlflow_model_version, artifact_checksum, tokenizer_hash,
                   context_limit, max_output_tokens, is_scratch_model, capability_note
            FROM model_revisions
            WHERE model_alias = ? AND readiness_status = 'READY'
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (rs, rowNum) -> new ModelCatalogEntry(
                rs.getString("model_alias"),
                rs.getString("revision_tag"),
                rs.getString("mlflow_model_version"),
                rs.getString("artifact_checksum"),
                rs.getString("tokenizer_hash"),
                rs.getInt("context_limit"),
                rs.getInt("max_output_tokens"),
                rs.getBoolean("is_scratch_model"),
                rs.getString("capability_note")
            ),
            cleanAlias
        );
        if (rows.isEmpty()) {
            return Optional.empty();
        }
        ModelCatalogEntry found = rows.getFirst();
        putInRedisQuietly(found);
        return Optional.of(found);
    }

    /**
     * Retrieve immutable metadata for a specific `revision_tag` via Redis cache-aside
     * (`model:rev:{revision_tag}`) with PostgreSQL fallback.
     */
    public Optional<ModelCatalogEntry> getByRevisionTag(String revisionTag) {
        if (revisionTag == null || revisionTag.isBlank()) {
            return Optional.empty();
        }
        String cleanTag = revisionTag.trim();
        String redisKey = REDIS_REVISION_KEY_PREFIX + cleanTag;

        try {
            String cachedJson = redisTemplate.opsForValue().get(redisKey);
            if (cachedJson != null && !cachedJson.isBlank()) {
                return Optional.of(objectMapper.readValue(cachedJson, ModelCatalogEntry.class));
            }
        } catch (Exception ignored) {
            // Fallback to PostgreSQL on Redis error or corrupt cache entry
        }

        List<ModelCatalogEntry> rows = jdbcTemplate.query(
            """
            SELECT model_alias, revision_tag, mlflow_model_version, artifact_checksum, tokenizer_hash,
                   context_limit, max_output_tokens, is_scratch_model, capability_note
            FROM model_revisions
            WHERE revision_tag = ?
            LIMIT 1
            """,
            (rs, rowNum) -> new ModelCatalogEntry(
                rs.getString("model_alias"),
                rs.getString("revision_tag"),
                rs.getString("mlflow_model_version"),
                rs.getString("artifact_checksum"),
                rs.getString("tokenizer_hash"),
                rs.getInt("context_limit"),
                rs.getInt("max_output_tokens"),
                rs.getBoolean("is_scratch_model"),
                rs.getString("capability_note")
            ),
            cleanTag
        );
        if (rows.isEmpty()) {
            return Optional.empty();
        }
        ModelCatalogEntry entry = rows.getFirst();
        putInRedisQuietly(entry);
        return Optional.of(entry);
    }

    public boolean isCachedInRedis(String revisionTag) {
        try {
            return Boolean.TRUE.equals(redisTemplate.hasKey(REDIS_REVISION_KEY_PREFIX + revisionTag));
        } catch (Exception ex) {
            return false;
        }
    }

    public void evictRevisionCache(String revisionTag, String modelAlias) {
        try {
            if (revisionTag != null) {
                redisTemplate.delete(REDIS_REVISION_KEY_PREFIX + revisionTag);
            }
            if (modelAlias != null) {
                redisTemplate.delete(REDIS_ALIAS_KEY_PREFIX + modelAlias);
            }
        } catch (Exception ignored) {
            // Ignore Redis eviction errors
        }
    }

    private void putInRedisQuietly(ModelCatalogEntry entry) {
        try {
            String json = objectMapper.writeValueAsString(entry);
            redisTemplate.opsForValue().set(
                REDIS_REVISION_KEY_PREFIX + entry.model_revision(),
                json,
                cacheTtl
            );
            redisTemplate.opsForValue().set(
                REDIS_ALIAS_KEY_PREFIX + entry.alias(),
                entry.model_revision(),
                cacheTtl
            );
        } catch (JsonProcessingException | RuntimeException ignored) {
            // Non-fatal: PostgreSQL remains authoritative
        }
    }
}

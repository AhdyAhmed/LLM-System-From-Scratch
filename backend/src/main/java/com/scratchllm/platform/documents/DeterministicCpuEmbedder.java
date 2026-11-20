package com.scratchllm.platform.documents;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Day073 & Day074: Deterministic document chunker (`paragraph-window-v1`) and pinned
 * 384-dimensional CPU text embedding component (`pretrained-cpu-hash384-v1`).
 *
 * IMPORTANT PROVENANCE NOTE:
 * This 384-dimensional CPU embedding component is used exclusively for `pgvector` exact
 * cosine similarity search and hybrid retrieval. It is explicitly separate from the
 * from-scratch causal decoder-only transformer (`ScratchGPT`).
 */
public final class DeterministicCpuEmbedder {

    public static final String EMBEDDING_MODEL_REVISION = "pretrained-cpu-hash384-v1";
    public static final int EMBEDDING_DIMENSION = 384;
    public static final String CHUNKER_VERSION = "paragraph-window-v1";

    private static final Pattern WORD_PATTERN = Pattern.compile("[a-z0-9]+");
    private static final Set<String> STOPWORDS = Set.of(
        "a", "an", "and", "are", "as", "at", "be", "by", "do", "does",
        "for", "from", "how", "in", "into", "is", "it", "of", "on", "or",
        "that", "the", "to", "was", "what", "when", "where", "which", "who", "why", "with"
    );

    private DeterministicCpuEmbedder() {}

    public record DocumentChunkSpec(
        int chunkOrder,
        String sourceLocation,
        String chunkText,
        String embeddingRevision,
        float[] embedding,
        String pgVectorLiteral
    ) {}

    public static String normalizeDocumentText(String rawText) {
        if (rawText == null) {
            return "";
        }
        return rawText
            .replace("\0", "")
            .replace("\r\n", "\n")
            .replace("\r", "\n")
            .trim();
    }

    public static List<DocumentChunkSpec> chunkDocumentText(
        String rawText,
        int documentRevision,
        int maxChunkChars,
        int overlapChars
    ) {
        String normalized = normalizeDocumentText(rawText);
        if (normalized.isEmpty()) {
            return List.of();
        }

        String[] rawParagraphs = normalized.split("\\n\\s*\\n");
        List<Span> rawSpans = new ArrayList<>();
        int cursor = 0;

        for (String p : rawParagraphs) {
            String para = p.trim();
            if (para.isEmpty()) {
                continue;
            }
            int idx = normalized.indexOf(para, cursor);
            if (idx < 0) {
                idx = cursor;
            }
            cursor = idx + para.length();

            if (para.length() <= maxChunkChars) {
                rawSpans.add(new Span(idx, idx + para.length(), para));
            } else {
                int step = Math.max(40, maxChunkChars - overlapChars);
                int pos = 0;
                while (pos < para.length()) {
                    int endPos = Math.min(para.length(), pos + maxChunkChars);
                    String window = para.substring(pos, endPos).trim();
                    if (!window.isEmpty()) {
                        rawSpans.add(new Span(idx + pos, idx + endPos, window));
                    }
                    if (pos + maxChunkChars >= para.length()) {
                        break;
                    }
                    pos += step;
                }
            }
        }

        List<Span> mergedSpans = new ArrayList<>();
        int i = 0;
        while (i < rawSpans.size()) {
            Span cur = rawSpans.get(i);
            if (i + 1 < rawSpans.size()
                && cur.text().length() < 60
                && cur.text().length() + 2 + rawSpans.get(i + 1).text().length() <= maxChunkChars) {
                Span next = rawSpans.get(i + 1);
                mergedSpans.add(new Span(cur.startChar(), next.endChar(), cur.text() + "\n\n" + next.text()));
                i += 2;
            } else {
                mergedSpans.add(cur);
                i += 1;
            }
        }

        List<DocumentChunkSpec> specs = new ArrayList<>(mergedSpans.size());
        for (int order = 0; order < mergedSpans.size(); order++) {
            Span span = mergedSpans.get(order);
            String sourceLoc = "rev" + documentRevision + ":chunk" + order + ":chars" + span.startChar() + "-" + span.endChar();
            float[] vec = embedText384(span.text());
            specs.add(new DocumentChunkSpec(
                order,
                sourceLoc,
                span.text(),
                EMBEDDING_MODEL_REVISION,
                vec,
                toPgVectorLiteral(vec)
            ));
        }
        return specs;
    }

    public static List<String> tokenizeRetrievalTerms(String text) {
        if (text == null || text.isBlank()) {
            return List.of();
        }
        Matcher matcher = WORD_PATTERN.matcher(text.toLowerCase(Locale.ROOT));
        List<String> terms = new ArrayList<>();
        while (matcher.find()) {
            String tok = matcher.group();
            if (!STOPWORDS.contains(tok)) {
                terms.add(lightStem(tok));
            }
        }
        return terms;
    }

    private static String lightStem(String word) {
        if (word.length() > 5 && word.endsWith("ing")) {
            return word.substring(0, word.length() - 3);
        }
        if (word.length() > 4 && word.endsWith("ed")) {
            return word.substring(0, word.length() - 2);
        }
        if (word.length() > 4 && word.endsWith("es")) {
            return word.substring(0, word.length() - 2);
        }
        if (word.length() > 3 && word.endsWith("s") && !word.endsWith("ss")) {
            return word.substring(0, word.length() - 1);
        }
        return word;
    }

    public static float[] embedText384(String text) {
        float[] vec = new float[EMBEDDING_DIMENSION];
        List<String> terms = tokenizeRetrievalTerms(text);
        if (terms.isEmpty()) {
            vec[0] = 1.0f;
            return vec;
        }

        List<FeatureWeight> features = new ArrayList<>();
        for (String t : terms) {
            features.add(new FeatureWeight("u:" + t, 2.0f));
            if (t.length() >= 4) {
                for (int j = 0; j <= t.length() - 4; j++) {
                    features.add(new FeatureWeight("c4:" + t.substring(j, j + 4), 0.5f));
                }
            }
        }
        for (int i = 0; i < terms.size() - 1; i++) {
            features.add(new FeatureWeight("b:" + terms.get(i) + "_" + terms.get(i + 1), 1.5f));
        }

        try {
            MessageDigest sha256 = MessageDigest.getInstance("SHA-256");
            for (FeatureWeight fw : features) {
                byte[] digest = sha256.digest(fw.feature().getBytes(StandardCharsets.UTF_8));
                int idx1 = (((digest[0] & 0xFF) << 8) | (digest[1] & 0xFF)) % EMBEDDING_DIMENSION;
                float sign1 = ((digest[2] & 1) == 0) ? 1.0f : -1.0f;
                int idx2 = (((digest[3] & 0xFF) << 8) | (digest[4] & 0xFF)) % EMBEDDING_DIMENSION;
                float sign2 = ((digest[5] & 1) == 0) ? 1.0f : -1.0f;
                vec[idx1] += sign1 * fw.weight();
                vec[idx2] += sign2 * (fw.weight() * 0.5f);
            }
        } catch (NoSuchAlgorithmException ex) {
            throw new IllegalStateException(ex);
        }

        double normSq = 0.0;
        for (float v : vec) {
            normSq += (double) v * v;
        }
        if (normSq <= 1e-12) {
            vec[0] = 1.0f;
            return vec;
        }
        double invNorm = 1.0 / Math.sqrt(normSq);
        for (int i = 0; i < EMBEDDING_DIMENSION; i++) {
            vec[i] = (float) (Math.round(vec[i] * invNorm * 1_000_000.0) / 1_000_000.0);
        }
        return vec;
    }

    public static String toPgVectorLiteral(float[] vec) {
        StringBuilder sb = new StringBuilder(vec.length * 9 + 2);
        sb.append('[');
        for (int i = 0; i < vec.length; i++) {
            if (i > 0) {
                sb.append(',');
            }
            sb.append(String.format(Locale.ROOT, "%.6f", vec[i]));
        }
        sb.append(']');
        return sb.toString();
    }

    private record Span(int startChar, int endChar, String text) {}
    private record FeatureWeight(String feature, float weight) {}
}

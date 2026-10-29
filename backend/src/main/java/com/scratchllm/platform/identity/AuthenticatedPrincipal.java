package com.scratchllm.platform.identity;

import java.security.Principal;
import java.util.UUID;

/**
 * Immutable authenticated principal populated from the shared Redis session store.
 */
public record AuthenticatedPrincipal(
    UUID userId,
    String email,
    String sessionId
) implements Principal {

    @Override
    public String getName() {
        return userId.toString();
    }
}

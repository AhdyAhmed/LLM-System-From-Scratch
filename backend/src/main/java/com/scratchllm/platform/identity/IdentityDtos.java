package com.scratchllm.platform.identity;

import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;
import java.util.List;
import java.util.UUID;

/**
 * Request and response DTOs for identity, registration, login, and workspace memberships.
 */
public final class IdentityDtos {

    private IdentityDtos() {}

    public record RegisterRequest(
        @NotBlank(message = "Email is required")
        @Email(message = "Must be a valid email address")
        @Size(max = 254, message = "Email must not exceed 254 characters")
        String email,

        @NotBlank(message = "Password is required")
        @Size(min = 12, max = 128, message = "Password must be between 12 and 128 characters")
        String password,

        @NotBlank(message = "Workspace name is required")
        @Size(min = 2, max = 80, message = "Workspace name must be between 2 and 80 characters")
        String workspace_name
    ) {}

    public record LoginRequest(
        @NotBlank(message = "Email is required")
        @Email(message = "Must be a valid email address")
        String email,

        @NotBlank(message = "Password is required")
        String password
    ) {}

    public record WorkspaceMembershipDto(
        UUID workspace_id,
        String name,
        String role
    ) {}

    public record UserProfileResponse(
        UUID user_id,
        String email,
        List<WorkspaceMembershipDto> workspaces
    ) {}
}

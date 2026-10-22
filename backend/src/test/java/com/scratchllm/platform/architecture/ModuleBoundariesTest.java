package com.scratchllm.platform.architecture;

import static org.assertj.core.api.Assertions.assertThat;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

import java.util.List;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.core.env.Environment;
import org.springframework.test.web.servlet.MockMvc;

/**
 * Day005 & Day007 module boundary and platform configuration verification test.
 */
@SpringBootTest
@AutoConfigureMockMvc
class ModuleBoundariesTest {

    @Autowired
    private Environment environment;

    @Autowired
    private MockMvc mockMvc;

    @Test
    void verifiesAllNineRequiredDomainPackagesExistAndOpenInViewIsDisabled() {
        assertThat(environment.getProperty("spring.jpa.open-in-view", Boolean.class))
            .isFalse();

        List<String> requiredPackages = List.of(
            "com.scratchllm.platform.identity",
            "com.scratchllm.platform.conversation",
            "com.scratchllm.platform.generation",
            "com.scratchllm.platform.usage",
            "com.scratchllm.platform.knowledge",
            "com.scratchllm.platform.modelcatalog",
            "com.scratchllm.platform.jobs",
            "com.scratchllm.platform.infrastructure",
            "com.scratchllm.platform.platform"
        );

        for (String pkgName : requiredPackages) {
            Package pkg = ClassLoader.getSystemClassLoader().getDefinedPackage(pkgName);
            if (pkg == null) {
                try {
                    Class.forName(pkgName + ".package-info");
                } catch (ClassNotFoundException ignored) {
                    // platform and generation have concrete classes loaded
                }
                pkg = ClassLoader.getSystemClassLoader().getDefinedPackage(pkgName);
            }
            assertThat(pkg).as("Package %s must be defined", pkgName).isNotNull();
        }
    }

    @Test
    void csrfBootstrapEndpointReturnsContractTokenStructure() throws Exception {
        mockMvc.perform(get("/auth/csrf"))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.header_name").value("X-CSRF-TOKEN"))
            .andExpect(jsonPath("$.parameter_name").value("_csrf"))
            .andExpect(jsonPath("$.token").isNotEmpty());
    }
}

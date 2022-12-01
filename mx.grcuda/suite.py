# Copyright (c) 2021, NECSTLab, Politecnico di Milano. All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NECSTLab nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#  * Neither the name of Politecnico di Milano nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

suite = {
    # --------------------------------------------------------------------------------------------------------------
    #
    #  METADATA
    #
    # --------------------------------------------------------------------------------------------------------------
    "mxversion": "5.190.1",
    "name": "grcuda",
    "versionConflictResolution": "latest",

    "version": "1.0.0",
    "release": False,
    "groupId": "com.nvidia.grcuda",

    "developer": {
        "name": "GrCUDA Developers",
        "organization": "GrCUDA Developers",
    },


    # --------------------------------------------------------------------------------------------------------------
    #
    #  DEPENDENCIES
    #
    # --------------------------------------------------------------------------------------------------------------
    "imports": {
        "suites": [
            {
                "name": "truffle",
                "version": "84541b16ae8a8726a0e7d76c7179d94a57ed84ee",
                "subdir": True,
                "urls": [
                    {"url": "https://github.com/oracle/graal", "kind": "git"},
                ]
            },
        ],
    },

    # --------------------------------------------------------------------------------------------------------------
    #
    #  REPOS
    #
    # --------------------------------------------------------------------------------------------------------------
    "repositories": {
    },

    "defaultLicense": "BSD-3",

    # --------------------------------------------------------------------------------------------------------------
    #
    #  LIBRARIES
    #
    # --------------------------------------------------------------------------------------------------------------
    "libraries": {
        "pmml4s_2.13" : {
            "urls" : ["https://repo1.maven.org/maven2/org/pmml4s/pmml4s_2.13/0.9.17/pmml4s_2.13-0.9.17-sources.jar"],
            "maven" : {
                "groupId" : "org.pmml4s",
                "artifactId" : "pmml4s_2.13",
                "version" : "0.9.17",
            },
            "sha1": "sha1=911867c618c821fc9b2af3d741eb20e5034bc897",
            "license" : ["Apache 2.0"],
            "dep": ["scala-library", "scala-xml_2.13", "commons-text", "spray-json_2.13", "commons-math3"]
        },
        "scala-xml_2.13" : {
            "urls" : ["https://repo1.maven.org/maven2/org/scala-lang/modules/scala-xml_2.13/2.1.0/scala-xml_2.13-2.1.0-sources.jar"],
            "maven" : {
                "groupId" : "org.scala-lang.modules",
                "artifactId" : "scala-xml_2.13",
                "version" : "2.1.0",
            },
            "sha1": "sha1=0fd264db1c7415552a8d2cb28ec65bcc922fc298",
            "license" : ["Apache 2.0"],
            "dep": ["scala-library"]
        },
        "scala-library" : {
            "urls" : ["https://repo1.maven.org/maven2/org/scala-lang/scala-library/2.13.10/scala-library-2.13.10-sources.jar"],
            "maven" : {
                "groupId" : "org.scala-lang",
                "artifactId" : "pmml4s_2.13",
                "version" : "2.13.10",
            },
            "sha1": "sha1=68e791105b904f26ee2eaa8c17e740c8076e36eb",
            "license" : ["Apache 2.0"]
        },
        "commons-text" : {
            "urls" : ["https://repo1.maven.org/maven2/org/apache/commons/commons-text/1.10.0/commons-text-1.10.0-sources.jar"],
            "maven" : {
                "groupId" : "org.apache.commons",
                "artifactId" : "commons-text",
                "version" : "1.10.0",
            },
            "sha1": "sha1=e8f5b9b64ff9aeaa2739f2eb4cfd6c48834a5df2",
            "license" : ["Apache 2.0"],
            "dep": ["commons-lang3"]
        },
        "commons-lang3" : {
            "urls" : ["https://repo1.maven.org/maven2/org/apache/commons/commons-lang3/3.12.0/commons-lang3-3.12.0-sources.jar"],
            "maven" : {
                "groupId" : "org.apache.commons",
                "artifactId" : "commons-lang3",
                "version" : "3.12.0",
            },
            "sha1": "sha1=5437944fa6d2c64e754c13f83ef0a315101f68fc",
            "license" : ["Apache 2.0"]
        },
        "commons-math3" : {
            "urls" : ["https://repo1.maven.org/maven2/org/apache/commons/commons-math3/3.6.1/commons-math3-3.6.1-sources.jar"],
            "maven" : {
                "groupId" : "org.apache.commons",
                "artifactId" : "commons-math3",
                "version" : "3.6.1",
            },
            "sha1": "sha1=8fab23986ea8886af34818daf32a718e81dc98ba",
            "license" : ["Apache 2.0"]
        },
        "spray-json_2.13" : {
            "urls" : ["https://repo1.maven.org/maven2/io/spray/spray-json_2.13/1.3.6/spray-json_2.13-1.3.6-sources.jar"],
            "maven" : {
                "groupId" : "io.spray",
                "artifactId" : "spray-json_2.13",
                "version" : "1.3.6",
            },
            "sha1": "sha1=6fb79573827c6d82ddf4882f76fdfdb542674cdc",
            "license" : ["Apache 2.0"],
            "dep": ["scala-library"]
        },
    },

    # --------------------------------------------------------------------------------------------------------------
    #
    #  PROJECTS
    #
    # --------------------------------------------------------------------------------------------------------------
    "externalProjects": {
    },


    "projects": {
        "com.nvidia.grcuda.parser.antlr": {
            "subDir": "projects",
            "buildEnv": {
                "ANTLR_JAR": "<path:truffle:ANTLR4_COMPLETE>",
                "PARSER_PATH": "<src_dir:com.nvidia.grcuda>/com/nvidia/grcuda/parser/antlr",
                "OUTPUT_PATH": "<src_dir:com.nvidia.grcuda>/com/nvidia/grcuda/parser/antlr",
                "PARSER_PKG": "com.nvidia.grcuda.parser.antlr",
                "POSTPROCESSOR": "<src_dir:com.nvidia.grcuda.parser.antlr>/postprocessor.py",
            },
            "dependencies": [
                "truffle:ANTLR4_COMPLETE",
            ],
            "native": True,
            "vpath": True,
        },
        "com.nvidia.grcuda": {
            "subDir": "projects",
            "license": ["BSD-3"],
            "sourceDirs": ["src"],
            "javaCompliance": "8+",
            "annotationProcessors": ["truffle:TRUFFLE_DSL_PROCESSOR"],
            "dependencies": [
                "truffle:TRUFFLE_API",
                "sdk:GRAAL_SDK",
                "truffle:ANTLR4",
            ],
            "buildDependencies": ["com.nvidia.grcuda.parser.antlr"],
            "checkstyleVersion": "8.8",
        },
        "com.nvidia.grcuda.test": {
            "subDir": "projects",
            "sourceDirs": ["src"],
            "dependencies": [
                "com.nvidia.grcuda",
                "mx:JUNIT",
                "truffle:TRUFFLE_TEST",
            ],
            "checkstyle": "com.nvidia.grcuda",
            "javaCompliance": "8+",
            "annotationProcessors": ["truffle:TRUFFLE_DSL_PROCESSOR"],
            "workingSets": "Truffle,CUDA",
            "testProject": True,
        },
    },

    "licenses": {
        "BSD-3": {
            "name": "3-Clause BSD License",
            "url": "http://opensource.org/licenses/BSD-3-Clause",
        },
        "Apache 2.0": {
            "name": "Apache 2.0 License",
            "url": "http://www.apache.org/licenses/LICENSE-2.0.txt",
        },
    },

    # --------------------------------------------------------------------------------------------------------------
    #
    #  DISTRIBUTIONS
    #
    # --------------------------------------------------------------------------------------------------------------
    "distributions": {
        "GRCUDA": {
            "dependencies": [
                "com.nvidia.grcuda",
            ],
            "distDependencies": [
                "truffle:TRUFFLE_API",
                "sdk:GRAAL_SDK",
            ],
            "sourcesPath": "grcuda.src.zip",
            "description": "GrCUDA",
            "javaCompliance": "8+",
        },

        "GRCUDA_UNIT_TESTS": {
            "description": "GrCUDA unit tests",
            "dependencies": [
                "com.nvidia.grcuda.test",
            ],
            "exclude": ["mx:JUNIT"],
            "distDependencies": [
                "GRCUDA",
                "truffle:TRUFFLE_TEST"
            ],
            "sourcesPath": "grcuda.tests.src.zip",
            "testDistribution": True,
            "javaCompliance": "8+",
        },
    },
}

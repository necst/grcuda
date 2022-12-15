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
            "maven" : {
                "groupId" : "org.pmml4s",
                "artifactId" : "pmml4s_2.13",
                "version" : "0.9.17",
            },
            "sha1" : "5ed6220264b215e9c740bfb2027ef03e5d500bcf",
            "license" : ["Apache 2.0"],
            "dep" : ["scala-library", "scala-xml_2.13", "commons-text", "spray-json_2.13", "commons-math3"]
        },
        "scala-xml_2.13" : {
            "maven" : {
                "groupId" : "org.scala-lang.modules",
                "artifactId" : "scala-xml_2.13",
                "version" : "1.2.0",
            },
            "sha1" : "f6abd60d28c189f05183b26c5363713d1d126b83",
            "license" : ["Apache 2.0"],
            "dep" : ["scala-library"]
        },
        "scala-library" : {
            "maven" : {
                "groupId" : "org.scala-lang",
                "artifactId" : "scala-library",
                "version" : "2.13.8",
            },
            "sha1" : "5a865f03a794b27e6491740c4c419a19e4511a3d",
            "license" : ["Apache 2.0"]
        },
        "commons-text" : {
            "maven" : {
                "groupId" : "org.apache.commons",
                "artifactId" : "commons-text",
                "version" : "1.6",
            },
            "sha1" : "ba72cf0c40cf701e972fe7720ae844629f4ecca2",
            "license" : ["Apache 2.0"],
            "dep" : ["commons-lang3"]
        },
        "commons-lang3" : {
            "maven" : {
                "groupId" : "org.apache.commons",
                "artifactId" : "commons-lang3",
                "version" : "3.8.1",
            },
            "sha1" : "6505a72a097d9270f7a9e7bf42c4238283247755",
            "license" : ["Apache 2.0"]
        },
        "commons-math3" : {
            "maven" : {
                "groupId" : "org.apache.commons",
                "artifactId" : "commons-math3",
                "version" : "3.6.1",
            },
            "sha1" : "e4ba98f1d4b3c80ec46392f25e094a6a2e58fcbf",
            "license" : ["Apache 2.0"]
        },
        "spray-json_2.13" : {
            "maven" : {
                "groupId" : "io.spray",
                "artifactId" : "spray-json_2.13",
                "version" : "1.3.5",
            },
            "sha1" : "34b3a148e57870e30b797a636d0ae8eb1a1fcc99",
            "license" : ["Apache 2.0"],
            "dep" : ["scala-library"]
        },


        "weka-stable" : {
            "maven" : {
                "groupId" : "nz.ac.waikato.cms.weka",
                "artifactId" : "weka-stable",
                "version" : "3.8.6",
            },
            "sha1" : "7e16bf3b95a17458da6729473d3a0344ef906a16",
            "license" : ["Gpl 3.0"],
            "dep" : ["jfilechooser-bookmarks","java-cup","java-cup-runtime","mtj",
                     "netlib-java", "jakarta.activation-api", "jakarta.xml.bind-api",
                     "arpack_combined_all", "bounce", "jaxb-runtime"]
        },
        "jfilechooser-bookmarks" : {
            "maven" : {
                "groupId" : "com.github.fracpete",
                "artifactId" : "jfilechooser-bookmarks",
                "version" : "0.1.6",
            },
            "sha1" : "cf79ff5ab7fba256282e546da97a8e7136e9258b",
            "license" : ["Gpl 3.0"],
            "dep" : ["jclipboardhelper"]
        },
        "jclipboardhelper" : {
            "maven" : {
                "groupId" : "com.github.fracpete",
                "artifactId" : "jclipboardhelper",
                "version" : "0.1.0",
            },
            "license" : ["Gpl 3.0"],
            "sha1" : "c962ca1ca1040ba8b7dd605ecc9c35c2aa19679a"
        },
        "java-cup" : {
            "maven" : {
                "groupId" : "com.github.vbmacher",
                "artifactId" : "java-cup",
                "version" : "11b-20160615",
            },
            "license" : ["CUP"],
            "sha1" : "589ba48d4cd8e35935178ba70a9b7911b062978b"
        },
        "java-cup-runtime" : {
            "maven" : {
                "groupId" : "com.github.vbmacher",
                "artifactId" : "java-cup-runtime",
                "version" : "11b-20160615",
            },
            "license" : ["CUP"],
            "sha1" : "d934da5580347e172a3a25a71fa86e1d3554520e"
        },
        "mtj" : {
            "maven" : {
                "groupId" : "com.googlecode.matrix-toolkits-java",
                "artifactId" : "mtj",
                "version" : "1.0.4",
            },
            "license" : ["GNU"],
            "sha1" : "e14ed840ff5e15de92dba2d1af29201fa70a0f35",
            #"dep" : ["all"]
        },
        #"all" : {
        #    "maven" : {
        #        "groupId" : "com.github.fommil.netlib",
        #        "artifactId" : "all",
        #        "version" : "1.1.2",
        #        #"modelVersion" : "4.0",
        #    },
        #    "license" : ["BSD-3"],
        #    "sha1" : "f235011206ac009adad2d6607f222649aba5ca9e",
        #},
        "netlib-java" : {
            "maven" : {
                "groupId" : "com.googlecode.netlib-java",
                "artifactId" : "netlib-java",
                "version" : "1.1",
            },
            "sha1" : "4a9c3b55b23430e5a6ad86101a88aea500ba8086",
            "license" : ["BSD-3"],
            "dep" : ["core"]
        },
        "core" : {
            "maven" : {
                "groupId" : "com.github.fommil.netlib",
                "artifactId" : "core",
                "version" : "1.1",
            },
            "sha1" : "839e9b8107e43741810e6cf66798ef15543e9f98",
            "license" : ["BSD-3"]
        },
        "istack-commons-runtime" : {
            "maven" : {
                "groupId" : "com.sun.istack",
                "artifactId" : "istack-commons-runtime",
                "version" : "3.0.12",
            },
            "sha1" : "cbbe1a62b0cc6c85972e99d52aaee350153dc530",
            "license" : ["Eclipse 1.0"]
        },
        "jakarta.activation-api" : {
            "maven" : {
                "groupId" : "jakarta.activation",
                "artifactId" : "jakarta.activation-api",
                "version" : "1.2.2",
            },
            "sha1" : "99f53adba383cb1bf7c3862844488574b559621f",
            "license" : ["EDL 1.0"]
        },
        "jakarta.xml.bind-api" : {
            "maven" : {
                "groupId" : "jakarta.xml.bind",
                "artifactId" : "jakarta.xml.bind-api",
                "version" : "2.3.3",
            },
            "sha1" : "48e3b9cfc10752fba3521d6511f4165bea951801",
            "license" : ["EDL 1.0"],
            "dep" : ["jakarta.activation-api"]
        },
        "arpack_combined_all" : {
            "maven" : {
                "groupId" : "net.sourceforge.f2j",
                "artifactId" : "arpack_combined_all",
                "version" : "0.1",
            },
            "sha1" : "225619a060b42605b4d9fd4af11815664abf26eb",
            "license" : ["BSD"]
        },
        "bounce" : {
            "maven" : {
                "groupId" : "nz.ac.waikato.cms.weka.thirdparty",
                "artifactId" : "bounce",
                "version" : "0.18",
            },
            "sha1" : "0d774f9a0aa5cbe08dea8907720c64c417aed861",
            "license" : ["BSD"]
        },
        "jaxb-runtime" : {
            "maven" : {
                "groupId" : "org.glassfish.jaxb",
                "artifactId" : "jaxb-runtime",
                "version" : "2.3.5",
            },
            "sha1" : "a169a961a2bb9ac69517ec1005e451becf5cdfab",
            "license" : ["Eclipse 1.0"],
            "dep" : ["istack-commons-runtime", "jakarta.xml.bind-api", "txw2"]
        },
        "txw2" : {
            "maven" : {
                "groupId" : "org.glassfish.jaxb",
                "artifactId" : "txw2",
                "version" : "2.3.5",
            },
            "sha1" : "ec8930fa62e7b1758b1664d135f50c7abe86a4a3",
            "license" : ["Eclipse 1.0"]
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
                "pmml4s_2.13",
                "scala-library",
                "scala-xml_2.13",
                "spray-json_2.13",
                "commons-math3",
                "commons-lang3",
                "commons-text",
                "weka-stable",
                "jfilechooser-bookmarks",
                "jclipboardhelper",
                "java-cup",
                "java-cup-runtime",
                "mtj",
                "netlib-java",
                "core",
                "istack-commons-runtime",
                "jakarta.activation-api",
                "jakarta.xml.bind-api",
                "arpack_combined_all",
                "bounce",
                "jaxb-runtime",
                "txw2"


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
        "Gpl 3.0": {
            "name": "GNU General Public License 3",
            "url": "http://www.gnu.org/licenses/gpl-3.0.txt",
        },
        "CUP": {
            "name": "CUP Parser Generator Copyright Notice, License, and Disclaimer",
            "url": "http://www2.cs.tum.edu/projects/cup/install.php",
        },
        "GNU": {
            "name": "GNU Lesser General Public License",
            "url": "http://www.gnu.org/licenses/lgpl.html"
        },
        "Eclipse 1.0": {
            "name": "Eclipse Distribution License - v 1.0",
            "url": "http://www.eclipse.org/org/documents/edl-v10.php"
        },
        "EDL 1.0": {
            "name": "EDL 1.0",
            "url": "http://www.eclipse.org/org/documents/edl-v10.php"
        },
        "BSD":{
            "name": "The BSD License",
            "url": "http://www.opensource.org/licenses/bsd-license.php"
        }
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

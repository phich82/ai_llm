import os
import platform
import subprocess

VISUAL_STUDIO_2022_TOOLS = "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\Common7\Tools\\VsDevCmd.bat"
VISUAL_STUDIO_2019_TOOLS = "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\BuildTools\\Common7\\Tools\\VsDevCmd.bat"
GCC_TOOLS = "C:\\msys64\\ucrt64\\bin\\g++.exe"

simple_cpp = """
#include <iostream>

int main() {
    std::cout << "Hello";
    return 0;
}
"""


class Excecutor:

    @classmethod
    def run(cls, cmd: str):
        try:
            run_result = subprocess.run(cmd, check=True, text=True, capture_output=True)
            return run_result.stdout if run_result.stdout else "SUCCESS"
        except:
            return ""

    @classmethod
    def compile_c(cls, filename_base):
        my_platform = platform.system()
        my_compiler = []

        try:
            with open("simple.cpp", "w") as f:
                f.write(simple_cpp)

            if my_platform == "Windows":
                if os.path.isfile(VISUAL_STUDIO_2022_TOOLS):
                    if os.path.isfile("./simple.exe"):
                        os.remove("./simple.exe")
                    compile_cmd = ["cmd", "/c", VISUAL_STUDIO_2022_TOOLS, "&", "cl", "simple.cpp"]
                    if cls.run(compile_cmd):
                        if cls.run(["./simple.exe"]) == "Hello":
                            my_compiler = ["Windows", "Visual Studio 2022", ["cmd", "/c", VISUAL_STUDIO_2022_TOOLS, "&", "cl", f"{filename_base}.cpp"]]

                if not my_compiler:
                    if os.path.isfile(VISUAL_STUDIO_2019_TOOLS):
                        if os.path.isfile("./simple.exe"):
                            os.remove("./simple.exe")
                        compile_cmd = ["cmd", "/c", VISUAL_STUDIO_2019_TOOLS, "&", "cl", "simple.cpp"]
                        if cls.run(compile_cmd):
                            if cls.run(["./simple.exe"]) == "Hello":
                                my_compiler = ["Windows", "Visual Studio 2019", ["cmd", "/c", VISUAL_STUDIO_2019_TOOLS, "&", "cl", f"{filename_base}.cpp"]]

                if os.path.isfile(GCC_TOOLS):
                    if os.path.isfile("./simple.exe"):
                        os.remove("./simple.exe")
                    compile_cmd = ["g++", "-g", "simple.cpp", "-o", "simple.exe"]
                    if cls.run(compile_cmd):
                        if cls.run(["./simple.exe"]) == "Hello":
                            my_compiler = ["Windows", "GCC (g++) MSYS64", ["g++", f"{filename_base}.cpp", "-o", f"{filename_base}.exe"]]

                if not my_compiler:
                    my_compiler = [my_platform, "Unavailable", []]

            elif my_platform == "Linux":
                if os.path.isfile("./simple"):
                    os.remove("./simple")
                compile_cmd = ["g++", "simple.cpp", "-o", "simple"]
                if cls.run(compile_cmd):
                    if cls.run(["./simple"]) == "Hello":
                        my_compiler = ["Linux", "GCC (g++)", ["g++", f"{filename_base}.cpp", "-o", f"{filename_base}" ]]

                if not my_compiler:
                    if os.path.isfile("./simple"):
                        os.remove("./simple")
                    compile_cmd = ["clang++", "simple.cpp", "-o", "simple"]
                    if cls.run(compile_cmd):
                        if cls.run(["./simple"]) == "Hello":
                            my_compiler = ["Linux", "Clang++", ["clang++", f"{filename_base}.cpp", "-o", f"{filename_base}"]]

                if not my_compiler:
                    my_compiler=[my_platform, "Unavailable", []]

            elif my_platform == "Darwin":
                if os.path.isfile("./simple"):
                    os.remove("./simple")
                compile_cmd = ["clang++", "-Ofast", "-std=c++17", "-march=armv8.5-a", "-mtune=apple-m1", "-mcpu=apple-m1", "-o", "simple", "simple.cpp"]
                if cls.run(compile_cmd):
                    if cls.run(["./simple"]) == "Hello":
                        my_compiler = ["Macintosh", "Clang++", ["clang++", "-Ofast", "-std=c++17", "-march=armv8.5-a", "-mtune=apple-m1", "-mcpu=apple-m1", "-o", f"{filename_base}", f"{filename_base}.cpp"]]

                if not my_compiler:
                    my_compiler=[my_platform, "Unavailable", []]
        except:
            my_compiler=[my_platform, "Unavailable", []]

        if my_compiler:
            return my_compiler
        else:
            return ["Unknown", "Unavailable", []]

compiler_cmd = Excecutor.compile_c("optimized")
print(compiler_cmd)
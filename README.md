### TRAK 25L 
Repository for the Real-Time Ray-Traced Soft Shadows project.

### Team members
* Paweł Dombrzalski
* Aleksander Kruk
* Bartosz Kisły

### Buliding the project 
* Open the project in Visual Studio
* Open CMakeLists.txt, save the file with `ctrl+s`, then build and run with `ctrl+f5`.
* Alternatively, you can build and run the project from the command line using CMake and NMake in release and debug mode:
```
cmake -S . -B out -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build out --config Release
./out/raytracer
```

```
cmake -S . -B out -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Debug
cmake --build out --config Debug
./out/raytracer
```

### Dependencies
* glm
* CUDA toolkit
* OpenGL - glfw, glad


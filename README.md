# legendary-waffle

# Setup
1. Get the vcpkg submodule `git submodule update --init vcpkg`
2. Follow setup steps from its [page](https://github.com/Microsoft/vcpkg#quick-start-windows)
    - `.\vcpkg\bootstrap-vcpkg.bat`
4. Install the two dependencies: OpenCV and Tesseract
    - `.\vcpkg\vcpkg.exe install opencv:x64-windows-static tesseract:x64-windows-static`

# Usage
`LegendaryWaffle.exe -show 1 -video "C:/path/to/video.mp4"`

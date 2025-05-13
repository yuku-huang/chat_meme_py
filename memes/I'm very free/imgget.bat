@echo off
setlocal enabledelayedexpansion

:: 設定輸出檔案
set "outfile=image_list.csv"
echo Filename > "%outfile%"

:: 依格式順序寫入檔案
for /R %%f in (*.jpeg) do (
    echo %%~nxf >> "%outfile%"
)

for /R %%f in (*.jpg) do (
    echo %%~nxf >> "%outfile%"
)

for /R %%f in (*.png) do (
    echo %%~nxf >> "%outfile%"
)

for /R %%f in (*.webp) do (
    echo %%~nxf >> "%outfile%"
)

echo 完成！已依格式順序輸出圖片檔名到 %outfile%
pause

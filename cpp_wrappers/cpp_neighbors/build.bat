@echo off
call conda activate KPConv
py setup.py build_ext --inplace
call conda deactivate

pause
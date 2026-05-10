import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# Don't try to collect the package __init__.py (which uses relative imports
# only valid when imported as 'comfyui_unividx' under ComfyUI).
collect_ignore = ["__init__.py"]

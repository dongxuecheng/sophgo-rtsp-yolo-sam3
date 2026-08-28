# Vendored SOPHON-DEMO YOLO26 runtime

Source: https://github.com/sophgo/sophon-demo/tree/dfe0dff31c21984b681536df9cb2b1ba6c96a8a4/sample/YOLO26/python

Included files:

- `yolo26_bmcv.py`

Only the YOLO26 inference class is retained. The standalone image/video demo,
drawing/saving helpers, COCO class/color utilities, unused imports and module-level
logging.basicConfig call were removed. The YOLO26 inference class itself is unchanged.
This module is imported by the service, not executed as a standalone CLI.

The bundled code is distributed under the included 2-Clause BSD license.

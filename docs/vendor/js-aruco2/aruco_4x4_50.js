/*
ArUco dictionary ARUCO_4X4_50 for js-aruco2 - the exact equivalent of
OpenCV's DICT_4X4_50 used by the ObjectLens Python pipeline.

Provenance: the 50 codes below are the first 50 entries of js-aruco2's
ARUCO_4X4_1000 dictionary (src/dictionaries/aruco_4x4_1000.js in
js-aruco2 2.0.0, itself extracted from OpenCV's
modules/aruco/src/predefined_dictionaries.hpp). They were verified
bit-for-bit against markers rendered by OpenCV's
cv2.aruco.generateImageMarker(DICT_4X4_50, id) for every id 0..49:
the code string equals the marker's inner 4x4 bits read row-major.
OpenCV's DICT_4X4_50 is likewise the first 50 codes of DICT_4X4_1000.
License: same 3-clause BSD license as OpenCV (see the js-aruco2
LICENSE file shipped alongside this folder's library files).
*/
AR.DICTIONARIES['ARUCO_4X4_50'] = {
  nBits: 16,
  tau: null,
  codeList: [[181,50],[15,154],[51,45],[153,70],[84,158],[121,205],[158,46],[196,242],[254,218],[207,86],[249,145],[17,167],[14,183],[42,15],[36,177],[38,62],[70,101],[102,0],[108,94],[118,175],[134,139],[176,43],[204,213],[221,130],[254,71],[148,113],[172,228],[165,84],[33,35],[52,111],[68,21],[87,178],[158,207],[240,203],[8,174],[9,41],[24,117],[4,255],[13,246],[28,90],[23,24],[42,40],[50,140],[56,178],[36,232],[46,235],[45,63],[75,100],[80,46],[80,19]]
};

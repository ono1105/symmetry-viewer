# Vendored Three.js

The Web viewer draws structures with [Three.js](https://threejs.org/) (MIT, see
`LICENSE`). Only the three files the server actually serves under
`/vendor/three/` are kept here, so neither Node.js nor `npm ci` is needed to run
or package the viewer.

| File | Taken from the `three` package |
|---|---|
| `three.module.js` | `build/three.module.js` |
| `three.core.js` | `build/three.core.js` |
| `addons/controls/TrackballControls.js` | `examples/jsm/controls/TrackballControls.js` |

Version: **0.185.1**.

One local change: `TrackballControls.js` imports `from 'three';`, a bare module
name that a browser cannot resolve without an import map. It is rewritten to
`from '/vendor/three/three.module.js';`. The server used to make this change on
every request; it is now done once, here.

To upgrade, fetch the package (for example `npm pack three@<version>` and
unpack it), copy the three files over these, apply the same one-line change,
and update the version above.

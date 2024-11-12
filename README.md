# motion-lines-demo - An executable

The `motion-lines-demo` executable is a <SUMMARY-OF-FUNCTIONALITY>.

Note that the `motion-lines-demo` executable in this package provides `build2` metadata.


## Usage

To start using `motion-lines-demo` in your project, add the following build-time
`depends` value to your `manifest`, adjusting the version constraint as
appropriate:

```
depends: * motion-lines-demo ^<VERSION>
```

Then import the executable in your `buildfile`:

```
import! [metadata] <TARGET> = motion-lines-demo%exe{<TARGET>}
```


## Importable targets

This package provides the following importable targets:

```
exe{<TARGET>}
```

<DESCRIPTION-OF-IMPORTABLE-TARGETS>


## Configuration variables

This package provides the following configuration variables:

```
[bool] config.motion_lines_demo.<VARIABLE> ?= false
```

<DESCRIPTION-OF-CONFIG-VARIABLES>

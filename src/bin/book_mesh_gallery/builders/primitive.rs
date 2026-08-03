use gaia::domain::core::scalar::Point3r;
use gaia::domain::geometry::primitives::{
    Antiprism, BiconcaveDisk, Capsule, Cone, Cube, Cuboctahedron, Cylinder, Disk, Dodecahedron,
    Elbow, Ellipsoid, FischerKochCySphere, FrdSphere, Frustum, GeodesicSphere, GyroidSphere,
    HelixSweep, Icosahedron, IwpSphere, LidinoidSphere, LinearSweep, NeoviusSphere, Octahedron,
    Pipe, PrimitiveMesh, Pyramid, RevolutionSweep, RoundedCube, SchwarzDSphere, SchwarzPSphere,
    SerpentineTube, SphericalShell, SplitPSphere, StadiumPrism, Tetrahedron, Torus,
    TruncatedIcosahedron, UvSphere,
};

use super::super::model::{GalleryResult, MeshCase};

fn primitive<P: PrimitiveMesh>(
    slug: &'static str,
    title: &'static str,
    source: &'static str,
    parameters: &'static str,
    builder: P,
) -> GalleryResult<MeshCase> {
    Ok(MeshCase {
        slug,
        title,
        source,
        parameters,
        mesh: builder.build()?,
    })
}

pub(crate) fn cases() -> GalleryResult<Vec<MeshCase>> {
    let mut cases = Vec::with_capacity(40);

    macro_rules! add_default {
        ($slug:literal, $title:literal, $source:literal, $builder:ident) => {
            cases.push(primitive(
                $slug,
                $title,
                $source,
                "Default builder parameters",
                <$builder>::default(),
            )?);
        };
    }

    add_default!(
        "tetrahedron",
        "Tetrahedron",
        "src/domain/geometry/primitives/tetrahedron.rs",
        Tetrahedron
    );
    add_default!(
        "cube",
        "Cube",
        "src/domain/geometry/primitives/cube.rs",
        Cube
    );
    cases.push(primitive(
        "uv-sphere",
        "UV sphere",
        "src/domain/geometry/primitives/sphere.rs",
        "radius=1, segments=24, stacks=12",
        UvSphere {
            radius: 1.0,
            center: Point3r::origin(),
            segments: 24,
            stacks: 12,
        },
    )?);
    add_default!(
        "cylinder",
        "Cylinder",
        "src/domain/geometry/primitives/cylinder.rs",
        Cylinder
    );
    add_default!(
        "cone",
        "Cone",
        "src/domain/geometry/primitives/cone.rs",
        Cone
    );
    add_default!(
        "torus",
        "Torus",
        "src/domain/geometry/primitives/torus.rs",
        Torus
    );
    cases.push(primitive(
        "linear-sweep",
        "Linear sweep",
        "src/domain/geometry/primitives/linear_sweep.rs",
        "regular hexagon profile, height=2",
        LinearSweep {
            profile: LinearSweep::regular_polygon(6, 1.0),
            height: 2.0,
        },
    )?);
    cases.push(primitive(
        "revolution-sweep",
        "Revolution sweep",
        "src/domain/geometry/primitives/revolution_sweep.rs",
        "rectangular profile, segments=24, angle=TAU",
        RevolutionSweep {
            profile: vec![(1.0, 0.0), (2.0, 0.0), (2.0, 0.5), (1.0, 0.5), (1.0, 0.0)],
            segments: 24,
            angle: std::f64::consts::TAU,
        },
    )?);
    add_default!(
        "octahedron",
        "Octahedron",
        "src/domain/geometry/primitives/octahedron.rs",
        Octahedron
    );
    add_default!(
        "icosahedron",
        "Icosahedron",
        "src/domain/geometry/primitives/icosahedron.rs",
        Icosahedron
    );
    add_default!(
        "ellipsoid",
        "Ellipsoid",
        "src/domain/geometry/primitives/ellipsoid.rs",
        Ellipsoid
    );
    add_default!(
        "frustum",
        "Frustum",
        "src/domain/geometry/primitives/frustum.rs",
        Frustum
    );
    add_default!(
        "capsule",
        "Capsule",
        "src/domain/geometry/primitives/capsule.rs",
        Capsule
    );
    add_default!(
        "pipe",
        "Pipe",
        "src/domain/geometry/primitives/pipe.rs",
        Pipe
    );
    add_default!(
        "elbow",
        "Elbow",
        "src/domain/geometry/primitives/elbow.rs",
        Elbow
    );
    add_default!(
        "biconcave-disk",
        "Biconcave disk",
        "src/domain/geometry/primitives/biconcave_disk.rs",
        BiconcaveDisk
    );
    add_default!(
        "disk",
        "Disk",
        "src/domain/geometry/primitives/disk.rs",
        Disk
    );
    add_default!(
        "spherical-shell",
        "Spherical shell",
        "src/domain/geometry/primitives/spherical_shell.rs",
        SphericalShell
    );
    add_default!(
        "stadium-prism",
        "Stadium prism",
        "src/domain/geometry/primitives/stadium_prism.rs",
        StadiumPrism
    );
    add_default!(
        "dodecahedron",
        "Dodecahedron",
        "src/domain/geometry/primitives/dodecahedron.rs",
        Dodecahedron
    );
    add_default!(
        "geodesic-sphere",
        "Geodesic sphere",
        "src/domain/geometry/primitives/geodesic_sphere.rs",
        GeodesicSphere
    );
    add_default!(
        "helix-sweep",
        "Helix sweep",
        "src/domain/geometry/primitives/helix_sweep.rs",
        HelixSweep
    );
    add_default!(
        "rounded-cube",
        "Rounded cube",
        "src/domain/geometry/primitives/rounded_cube.rs",
        RoundedCube
    );
    add_default!(
        "cuboctahedron",
        "Cuboctahedron",
        "src/domain/geometry/primitives/cuboctahedron.rs",
        Cuboctahedron
    );
    add_default!(
        "pyramid",
        "Pyramid",
        "src/domain/geometry/primitives/pyramid.rs",
        Pyramid
    );
    add_default!(
        "antiprism",
        "Antiprism",
        "src/domain/geometry/primitives/antiprism.rs",
        Antiprism
    );
    add_default!(
        "truncated-icosahedron",
        "Truncated icosahedron",
        "src/domain/geometry/primitives/truncated_icosahedron.rs",
        TruncatedIcosahedron
    );
    add_default!(
        "serpentine-tube",
        "Serpentine tube",
        "src/domain/geometry/primitives/serpentine_tube.rs",
        SerpentineTube
    );

    macro_rules! add_tpms {
        ($slug:literal, $title:literal, $source:literal, $builder:ident) => {
            cases.push(primitive(
                $slug,
                $title,
                $source,
                "radius=2, period=2, resolution=18, iso_value=0",
                $builder {
                    radius: 2.0,
                    period: 2.0,
                    resolution: 18,
                    iso_value: 0.0,
                },
            )?);
        };
    }

    add_tpms!(
        "gyroid-sphere",
        "Gyroid sphere",
        "src/domain/geometry/primitives/gyroid_sphere.rs",
        GyroidSphere
    );
    add_tpms!(
        "schwarz-p-sphere",
        "Schwarz P sphere",
        "src/domain/geometry/primitives/schwarz_p_sphere.rs",
        SchwarzPSphere
    );
    add_tpms!(
        "schwarz-d-sphere",
        "Schwarz D sphere",
        "src/domain/geometry/primitives/schwarz_d_sphere.rs",
        SchwarzDSphere
    );
    add_tpms!(
        "neovius-sphere",
        "Neovius sphere",
        "src/domain/geometry/primitives/neovius_sphere.rs",
        NeoviusSphere
    );
    add_tpms!(
        "lidinoid-sphere",
        "Lidinoid sphere",
        "src/domain/geometry/primitives/lidinoid_sphere.rs",
        LidinoidSphere
    );
    add_tpms!(
        "iwp-sphere",
        "I-WP sphere",
        "src/domain/geometry/primitives/iwp_sphere.rs",
        IwpSphere
    );
    add_tpms!(
        "split-p-sphere",
        "Split P sphere",
        "src/domain/geometry/primitives/split_p_sphere.rs",
        SplitPSphere
    );
    add_tpms!(
        "frd-sphere",
        "FRD sphere",
        "src/domain/geometry/primitives/frd_sphere.rs",
        FrdSphere
    );
    add_tpms!(
        "fischer-koch-cy-sphere",
        "Fischer–Koch C(Y) sphere",
        "src/domain/geometry/primitives/fischer_koch_cy_sphere.rs",
        FischerKochCySphere
    );

    Ok(cases)
}

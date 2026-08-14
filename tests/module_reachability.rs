use std::collections::{BTreeSet, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};

#[test]
fn every_source_module_is_reachable_from_a_crate_root() {
    let source_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let reachable = match reachable_modules(&source_root) {
        Ok(reachable) => reachable,
        Err(error) => panic!("module reachability scan failed: {error}"),
    };
    let mut source_files = BTreeSet::new();
    if let Err(error) = collect_rust_files(&source_root, &mut source_files) {
        panic!("source enumeration failed: {error}");
    }

    let orphans: Vec<_> = source_files.difference(&reachable).collect();
    assert!(
        orphans.is_empty(),
        "unreachable Rust source files: {}",
        orphans
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    );
}

fn reachable_modules(source_root: &Path) -> Result<BTreeSet<PathBuf>, String> {
    let roots = crate_roots(source_root)?;
    let mut reachable = BTreeSet::new();
    let mut pending: VecDeque<_> = roots.iter().cloned().collect();

    while let Some(module_file) = pending.pop_front() {
        if !reachable.insert(module_file.clone()) {
            continue;
        }

        let source = fs::read_to_string(&module_file)
            .map_err(|error| format!("{}: {error}", module_file.display()))?;
        for declaration in module_declarations(&source) {
            let child = resolve_module(&module_file, declaration)?;
            pending.push_back(child);
        }
    }

    Ok(reachable)
}

fn crate_roots(source_root: &Path) -> Result<Vec<PathBuf>, String> {
    let mut roots = Vec::new();
    for entry in
        fs::read_dir(source_root).map_err(|error| format!("{}: {error}", source_root.display()))?
    {
        let entry = entry.map_err(|error| format!("{}: {error}", source_root.display()))?;
        let path = entry.path();
        if path.is_file()
            && path
                .file_name()
                .is_some_and(|name| name == "lib.rs" || name == "main.rs")
        {
            roots.push(path);
        }
    }
    let bin_directory = source_root.join("bin");
    if bin_directory.is_dir() {
        for entry in fs::read_dir(&bin_directory)
            .map_err(|error| format!("{}: {error}", bin_directory.display()))?
        {
            let entry = entry.map_err(|error| format!("{}: {error}", bin_directory.display()))?;
            let path = entry.path();
            if path.is_file() && path.extension().is_some_and(|extension| extension == "rs") {
                roots.push(path);
            }
        }
    }
    roots.sort();
    Ok(roots)
}

fn collect_rust_files(path: &Path, files: &mut BTreeSet<PathBuf>) -> Result<(), String> {
    for entry in fs::read_dir(path).map_err(|error| format!("{}: {error}", path.display()))? {
        let entry = entry.map_err(|error| format!("{}: {error}", path.display()))?;
        let child = entry.path();
        if child.is_dir() {
            collect_rust_files(&child, files)?;
        } else if child.extension().is_some_and(|extension| extension == "rs") {
            files.insert(child);
        }
    }
    Ok(())
}

#[derive(Debug)]
struct ModuleDeclaration {
    name: String,
    path: Option<PathBuf>,
}

fn module_declarations(source: &str) -> Vec<ModuleDeclaration> {
    let mut declarations = Vec::new();
    let mut path_attribute = None;
    for line in source.lines() {
        let trimmed = line.trim();
        if let Some(path) = trimmed
            .strip_prefix("#[path = \"")
            .and_then(|value| value.strip_suffix("\"]"))
        {
            path_attribute = Some(PathBuf::from(path));
            continue;
        }
        let Some(module) = trimmed.strip_suffix(';') else {
            path_attribute = None;
            continue;
        };
        let mut words = module.split_whitespace();
        let mut name = None;
        while let Some(word) = words.next() {
            if word == "mod" {
                name = words.next();
                break;
            }
        }
        if let Some(name) = name {
            declarations.push(ModuleDeclaration {
                name: name.to_owned(),
                path: path_attribute.take(),
            });
        } else {
            path_attribute = None;
        }
    }
    declarations
}

fn resolve_module(parent: &Path, declaration: ModuleDeclaration) -> Result<PathBuf, String> {
    let parent_directory = parent
        .parent()
        .ok_or_else(|| format!("module has no parent: {}", parent.display()))?;
    let is_crate_root = parent
        .file_stem()
        .is_some_and(|stem| stem == "lib" || stem == "main")
        || parent_directory
            .file_name()
            .is_some_and(|name| name == "bin");
    let module_directory =
        if is_crate_root || parent.file_name().is_some_and(|name| name == "mod.rs") {
            parent_directory.to_owned()
        } else {
            parent_directory.join(
                parent
                    .file_stem()
                    .ok_or_else(|| format!("module has no stem: {}", parent.display()))?,
            )
        };
    let candidate = declaration.path.map_or_else(
        || module_directory.join(&declaration.name),
        |path| parent_directory.join(path),
    );
    let candidates = if candidate.extension().is_some() {
        vec![candidate]
    } else {
        vec![candidate.with_extension("rs"), candidate.join("mod.rs")]
    };
    candidates
        .into_iter()
        .find(|path| path.is_file())
        .ok_or_else(|| {
            format!(
                "module `{}` declared by {} has no source file",
                declaration.name,
                parent.display()
            )
        })
}

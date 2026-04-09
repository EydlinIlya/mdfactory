#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

params.csv_file = null
params.output_dir = "./results"

// Process 1: Generate folders and YAML files
process GENERATE_FOLDERS {
    input:
    path csv_file
    
    output:
    stdout emit: output_info
    
    script:
    // Resolve output directory relative to launch directory, not work directory
    def launch_dir = workflow.launchDir
    def output_path = params.output_dir.startsWith('/') ? params.output_dir : "${launch_dir}/${params.output_dir}"
    """
    ABSOLUTE_OUTPUT="${output_path}"
    echo "Creating output in: \$ABSOLUTE_OUTPUT" >&2
    
    mkdir -p "\$ABSOLUTE_OUTPUT"
    mdfactory prepare-build ${csv_file} --output "\$ABSOLUTE_OUTPUT"
    
    echo "Files created:" >&2
    ls -la "\$ABSOLUTE_OUTPUT" >&2
    
    echo "\$ABSOLUTE_OUTPUT"
    """
}

// Process 2: Read system directories from summary YAML
process READ_SYSTEM_DIRECTORIES {
    input:
    val output_dir
    val summary_yaml

    output:
    stdout emit: system_dirs

    script:
    """
    python3 << 'EOF'
import yaml

with open('${output_dir}/${summary_yaml}', 'r') as f:
    data = yaml.safe_load(f)

for dir_path in data.get('system_directory', []):
    print(dir_path)
EOF
    """
}

// Process 3: Build each system
process BUILD_FILES {
    input:
    val system_dir
    
    output:
    val system_dir
    
    script:
    """
    hash_name=\$(basename "${system_dir}")
    cd "${system_dir}"
    mdfactory build "\${hash_name}.yaml"
    """
}

workflow {
    if (!params.csv_file) {
        error "Please provide --csv_file parameter"
    }

    csv_ch = Channel.fromPath(params.csv_file, checkIfExists: true)
    summary_yaml = file(params.csv_file).baseName + ".yaml"

    GENERATE_FOLDERS(csv_ch)

    output_dir = GENERATE_FOLDERS.out.output_info.map { it.trim() }
    READ_SYSTEM_DIRECTORIES(output_dir, summary_yaml)
    
    system_dirs_ch = READ_SYSTEM_DIRECTORIES.out.system_dirs
        .splitText()
        .map { it.trim() }
        .filter { it && it != "" }
    
    BUILD_FILES(system_dirs_ch)
}

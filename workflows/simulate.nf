#!/usr/bin/env nextflow

nextflow.enable.dsl = 2

// Parameters
params.config_yaml = "config.yaml"
params.base_dir = "."

// Read YAML configuration
def readYamlConfig(yamlFile) {
    def yaml = new org.yaml.snakeyaml.Yaml()
    def config = yaml.load(new File(yamlFile).text)
    return config.hash
}

workflow {
    // Read the YAML file and create channel
    hash_list = readYamlConfig(params.config_yaml)
    hash_channel = Channel.fromList(hash_list)
    
    // Create input channel with all required files for minimization
    min_input_ch = hash_channel.map { hash ->
        tuple(
            hash,
            file("${params.base_dir}/${hash}/em.mdp"),
            file("${params.base_dir}/${hash}/system.pdb"),
            file("${params.base_dir}/${hash}/topology.top")
        )
    }
    
    // Run the GROMACS chain for each hash
    minimization(min_input_ch)
    nvt_equilibration(minimization.out)
    npt_equilibration(nvt_equilibration.out)
    production(npt_equilibration.out)
}

process minimization {
    tag "$hash"
    
    publishDir "${params.base_dir}/${hash}", mode: 'copy', overwrite: true, pattern: "min.*"
    
    input:
    tuple val(hash), path(em_mdp), path(system_pdb), path(topology_top)
    
    output:
    tuple val(hash), path("min.gro"), path("min.log"), emit: min_results
    
    script:
    """
    # Files are already staged by Nextflow - use them directly
    gmx grompp -f ${em_mdp} -c ${system_pdb} -p ${topology_top} -o min.tpr -maxwarn 1
    gmx mdrun -deffnm min -nt 12
    """
}

process nvt_equilibration {
    tag "$hash"
    
    publishDir "${params.base_dir}/${hash}", mode: 'copy', overwrite: true, pattern: "nvt.*"
    
    input:
    tuple val(hash), path(min_gro), path(min_log)
    
    output:
    tuple val(hash), path("nvt.gro"), path("nvt.cpt"), path("nvt.log"), emit: nvt_results
    
    script:
    def absolute_base_dir = file(params.base_dir).toAbsolutePath()
    """
    # Stage additional required files
    ln -s ${absolute_base_dir}/${hash}/nvt.mdp .
    ln -s ${absolute_base_dir}/${hash}/topology.top .
    
    # Preprocessing for NVT
    gmx grompp -f nvt.mdp -c ${min_gro} -r ${min_gro} -p topology.top -o nvt.tpr -maxwarn 1
    
    # Run NVT equilibration
    gmx mdrun -deffnm nvt -nt 12
    """
}

process npt_equilibration {
    tag "$hash"
    
    publishDir "${params.base_dir}/${hash}", mode: 'copy', overwrite: true, pattern: "npt.*"
    
    input:
    tuple val(hash), path(nvt_gro), path(nvt_cpt), path(nvt_log)
    
    output:
    tuple val(hash), path("npt.gro"), path("npt.cpt"), path("npt.log"), emit: npt_results
    
    script:
    def absolute_base_dir = file(params.base_dir).toAbsolutePath()
    """
    # Stage additional required files
    ln -s ${absolute_base_dir}/${hash}/npt.mdp .
    ln -s ${absolute_base_dir}/${hash}/topology.top .
    
    # Preprocessing for NPT
    gmx grompp -f npt.mdp -c ${nvt_gro} -r ${nvt_gro} -t ${nvt_cpt} -p topology.top -o npt.tpr -maxwarn 2
    
    # Run NPT equilibration
    gmx mdrun -deffnm npt -nt 12
    """
}

process production {
    tag "$hash"
    
    publishDir "${params.base_dir}/${hash}", mode: 'copy', overwrite: true, pattern: "prod.*"
    
    input:
    tuple val(hash), path(npt_gro), path(npt_cpt), path(npt_log)
    
    output:
    tuple val(hash), path("prod.gro"), path("prod.xtc"), path("prod.cpt"), path("prod.log"), path("prod.edr"), emit: prod_results
    
    script:
    def absolute_base_dir = file(params.base_dir).toAbsolutePath()
    """
    # Stage additional required files
    ln -s ${absolute_base_dir}/${hash}/md.mdp .
    ln -s ${absolute_base_dir}/${hash}/topology.top .
    
    # Preprocessing for production
    gmx grompp -f md.mdp -c ${npt_gro} -t ${npt_cpt} -p topology.top -o prod.tpr -maxwarn 2
    
    # Run production simulation
    gmx mdrun -deffnm prod -nt 12 -v
    """
}

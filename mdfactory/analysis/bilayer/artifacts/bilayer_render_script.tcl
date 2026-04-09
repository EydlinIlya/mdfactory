#!/usr/bin/tclsh
# Enhanced VMD script with high-quality Tachyon rendering
# Usage: vmd -dispdev text -e bilayer_render_script.tcl -args input.pdb trajectory.xtc output_prefix
package require pbctools
color change rgb 19 0.36863 0.73333 0.79216
color change rgb 26 0.58039 0.84314 0.87843
color change rgb 20 0.85098 0.29804 0.58039
color change rgb 27 0.89412 0.56078 0.74118
color change rgb 21 0.29804 0.20000 0.54902
color change rgb 28 0.54118 0.45098 0.78431
color change rgb 22 0.18824 0.40392 0.66667
# Next 7 colors medium color
color change rgb 29 0.40000 0.65490 0.91373
color change rgb 23 0.67843 0.80000 0.38824
color change rgb 30 0.80784 0.87843 0.61569
color change rgb 24 0.96471 0.79216 0.32941
color change rgb 31 0.98039 0.87451 0.56078
color change rgb 25 0.28235 0.60000 0.39216
color change rgb 32 0.48627 0.89804 0.65882

set pdb_file [lindex $argv 0]
set trajectory_file [lindex $argv 1]
set output_prefix [lindex $argv 2]

if {$pdb_file == "" || $trajectory_file == "" || $output_prefix == ""} {
    puts "Usage: vmd -dispdev text -e bilayer_render_script.tcl -args input.pdb trajectory.xtc output_prefix"
    exit
}

# Load the structure and trajectory
mol load pdb $pdb_file
set molid [molinfo top]
mol addfile $trajectory_file waitfor all

set total_frames [molinfo $molid get numframes]
if {$total_frames > 0} {
    set last_frame [expr $total_frames - 1]
    animate goto $last_frame
}

# Delete default representation
mol delrep 0 $molid

# Setup AOChalky material with custom outline
material change outline AOChalky 2.5
material change outlinewidth AOChalky 0.2

# Define selections and colors
set selections {
    {"resname ILN and noh" "ColorID 19"}
    {"resname ILP and noh" "ColorID 20"}
    {"resname HL and noh" "ColorID 21"}
    {"resname CHL and noh" "ColorID 23"}
}

# Add representations for each selection
foreach sel $selections {
    set selection [lindex $sel 0]
    set color [lindex $sel 1]
    
    mol representation Licorice
    mol color $color
    mol selection $selection
    mol material AOChalky
    mol addrep $molid
    
    puts "Added representation: $selection with color $color"
}

# Display settings
color Display Background white
display projection orthographic
#display rendermode GLSL
axes location off

# Turn on PBC box
#pbc box -toggle

# Lighting and effects
light 0 on
light 1 on
light 2 on
light 3 off

# Ambient occlusion and shadows
display ambientocclusion on
display aoambient 0.8
display aodirect 0.3
display shadows on
display antialias on

# Set Tachyon path: try TACHYON_PATH env var first, then search PATH
if {[info exists env(TACHYON_PATH)]} {
    set tachyon_path $env(TACHYON_PATH)
} else {
    if {[catch {set tachyon_path [exec which tachyon_LINUXAMD64]} err]} {
        puts "Error: Tachyon not found. Set TACHYON_PATH or add tachyon_LINUXAMD64 to PATH."
        exit 1
    }
}

# Render top view
puts "Rendering side view..."
display resetview
rotate x by 90
scale by 1.2

# Render to Tachyon format
render Tachyon ${output_prefix}_top.dat

# Convert with high quality settings
puts "Converting side view with Tachyon..."
set cmd [list $tachyon_path ${output_prefix}_top.dat -format TARGA -o ${output_prefix}_side.tga -res 2000 2000  -aasamples 12 -rescale_lights 0.4]
if {[catch {eval exec $cmd} result]} {
    puts "Error converting sid view: $result"
    puts "Command was: $cmd"
} else {
    puts "✓ Side view converted successfully"
    file delete ${output_prefix}_top.dat
}

# Render side view  
puts "Rendering top view..."
display resetview
scale by 1.2

# Render to Tachyon format
render Tachyon ${output_prefix}_side.dat

# Convert with high quality settings
puts "Converting top view with Tachyon..."
set cmd [list $tachyon_path ${output_prefix}_side.dat -format TARGA -o ${output_prefix}_top.tga -res 2000 2000 -aasamples 12 -rescale_lights 0.4]
if {[catch {eval exec $cmd} result]} {
    puts "Error converting top view: $result"
    puts "Command was: $cmd"
} else {
    puts "✓ Top view converted successfully"
    file delete ${output_prefix}_side.dat
}

puts "=== Rendering Summary ==="
if {[file exists ${output_prefix}_top.tga]} {
    puts "✓ Top view: ${output_prefix}_top.tga"
} else {
    puts "✗ Top view failed"
}

if {[file exists ${output_prefix}_side.tga]} {
    puts "✓ Side view: ${output_prefix}_side.tga"
} else {
    puts "✗ Side view failed"
}

# Clean up
mol delete $molid
quit

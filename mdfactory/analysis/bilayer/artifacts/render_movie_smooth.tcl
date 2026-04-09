#!/usr/bin/tclsh
# Enhanced VMD script with high-quality TachyonInternal rendering - Side View Movie
# With frame skipping and mol smoothrep smoothing
# Usage: vmd -dispdev text -e bilayer_movie_render.tcl -args input.pdb trajectory.dcd output_prefix [frame_step] [max_frames]
package require pbctools

# Custom color definitions (same as original)
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
set frame_step [lindex $argv 3]
set max_frames [lindex $argv 4]

if {$pdb_file == "" || $trajectory_file == "" || $output_prefix == ""} {
    puts "Usage: vmd -dispdev text -e bilayer_movie_render.tcl -args input.pdb trajectory.dcd output_prefix [frame_step] [max_frames]"
    exit
}

if {$frame_step == ""} {
    set frame_step 2
}

if {$max_frames == ""} {
    set max_frames 0
}

# Load the structure and trajectory
mol load pdb $pdb_file
set molid [molinfo top]
mol addfile $trajectory_file waitfor all

puts "Loaded structure: $pdb_file"
puts "Loaded trajectory: $trajectory_file"
set total_frames [molinfo $molid get numframes]
puts "Total frames in trajectory: $total_frames"

set render_frames [expr ($total_frames + $frame_step - 1) / $frame_step]
if {$max_frames > 0 && $render_frames > $max_frames} {
    set render_frames $max_frames
}
puts "Frames to render (step=$frame_step, max=$max_frames): $render_frames"

# Delete default representation
mol delrep 0 $molid

# Setup AOChalky material with custom outline (same as original)
material change outline AOChalky 2.5
material change outlinewidth AOChalky 0.2

# Define selections and colors (same as original)
set selections {
    {"resname ILN and noh" "ColorID 19"}
    {"resname ILP and noh" "ColorID 20"} 
    {"resname HL and noh" "ColorID 21"}
    {"resname CHL and noh" "ColorID 23"}
}

# Add representations for each selection (same as original)
set rep_id 0
foreach sel $selections {
    set selection [lindex $sel 0]
    set color [lindex $sel 1]
    
    mol representation Licorice
    mol color $color
    mol selection $selection
    mol material AOChalky
    mol addrep $molid
    
    # Apply smoothing to this representation (4-frame window)
    mol smoothrep $molid $rep_id 4
    puts "Added representation $rep_id: $selection with color $color (4-frame smoothing enabled)"
    
    incr rep_id
}

# Display settings (same as original)
color Display Background white
display projection orthographic
display rendermode Normal
axes location off

# Turn on PBC box
pbc box -toggle

# Lighting and effects (same as original)
light 0 on
light 1 on
light 2 on
light 3 off

# Ambient occlusion and shadows (same as original)
display ambientocclusion on
display aoambient 0.8
display aodirect 0.3
display shadows on
display antialias on

# Configure TachyonInternal with high-quality settings
render options TachyonInternal "-aasamples 12 -res 1000 1000 -rescale_lights 0.4"

# Set up side view orientation
puts "Setting up side view orientation..."
display resetview
# Rotate by 90 degrees around X-axis for side view of bilayer
rotate x by 90
scale by 1.5

# Create output directory for frames
set output_dir "${output_prefix}_frames"
file mkdir $output_dir

# Movie rendering loop - frame stepping
puts "Starting movie rendering with frame skipping and mol smoothrep smoothing..."
puts "Output directory: $output_dir"
puts "Smoothing: 4-frame window applied to all representations"

set output_frame 0
set max_source_frames [expr $render_frames * $frame_step]
for {set frame 0} {$frame < $total_frames && $frame < $max_source_frames} {set frame [expr $frame + $frame_step]} {
    # Go to current frame
    animate goto $frame
    
    # Update display to ensure proper rendering with smoothing
    display update
    
    # Generate frame filename with zero-padding
    set frame_file [format "%s/frame_%05d.tga" $output_dir $output_frame]
    
    # Render current frame
    render TachyonInternal $frame_file
    
    # Progress indicator
    if {[expr $output_frame % 5] == 0 || $frame >= [expr $total_frames - 2]} {
        set percent [expr int(100.0 * $frame / ($total_frames - 1))]
        puts "Rendered frame $output_frame (source frame $frame) - $percent% complete"
    }
    
    # Check if frame was rendered successfully
    if {![file exists $frame_file]} {
        puts "Warning: Frame $output_frame failed to render"
    }
    
    incr output_frame
}

puts ""
puts "=== Movie Rendering Complete ==="
puts "Total source frames: $total_frames"
puts "Frames rendered: $output_frame (step $frame_step)"
puts "Frame directory: $output_dir"
puts "Resolution: 2000x2000 pixels"
puts "Anti-aliasing: 12 samples"
puts "Smoothing: mol smoothrep with 4-frame window"
puts ""
puts "To create movie with ffmpeg (recommended framerate for skipped frames):"
puts "ffmpeg -framerate 15 -i ${output_dir}/frame_%05d.tga -c:v libx264 -pix_fmt yuv420p ${output_prefix}_movie.mp4"
puts ""
puts "Alternative high-quality encoding:"
puts "ffmpeg -framerate 15 -i ${output_dir}/frame_%05d.tga -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p ${output_prefix}_hq_movie.mp4"
puts ""
puts "For smoother playback (interpolated to 30fps):"
puts "ffmpeg -framerate 15 -i ${output_dir}/frame_%05d.tga -vf \"minterpolate=fps=30:mi_mode=mci\" -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p ${output_prefix}_smooth_30fps.mp4"

# Clean up
mol delete $molid
quit

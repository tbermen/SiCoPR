def save_cmd_line(config_file, chdata, num_fext, num_next, cli_name):
    """Build command-line string for running COM interactively (MATLAB lines 11165-11174)."""
    cmd_str = f"{cli_name}('{config_file}',{num_fext}, {num_next},'{chdata[0].filename}'"
    for i in range(1, len(chdata)):
        cmd_str += f",'{chdata[i].filename}'"
    cmd_str += ')'
    return cmd_str

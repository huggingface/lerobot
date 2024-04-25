PYPROJECT = "pyproject.toml"
DEPS = {
    "gym-pusht": '{ git = "git@github.com:huggingface/gym-pusht.git", optional = true}',
    "gym-xarm": '{ git = "git@github.com:huggingface/gym-xarm.git", optional = true}',
    "gym-aloha": '{ git = "git@github.com:huggingface/gym-aloha.git", optional = true}',
}


def update_envs_as_path_dependencies():
    with open(PYPROJECT) as file:
        lines = file.readlines()

    new_lines = []
    for line in lines:
        if any(dep in line for dep in DEPS.values()):
            for dep in DEPS:
                if dep in line:
                    new_line = f'{dep} = {{ path = "envs/{dep}/", optional = true}}\n'
                    new_lines.append(new_line)
                    break

        else:
            new_lines.append(line)

    with open(PYPROJECT, "w") as file:
        file.writelines(new_lines)


if __name__ == "__main__":
    update_envs_as_path_dependencies()

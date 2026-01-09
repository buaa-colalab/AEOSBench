SHELL := zsh

current_todd_version := $$(cat .todd_version)
latest_todd_version := $(shell curl -H "Accept: application/vnd.github.sha" -s https://api.github.com/repos/LutingWang/todd/commits/main)

define install_todd
	pipenv run pip uninstall -y todd_ai
	GIT_LFS_SKIP_SMUDGE=1 pipenv run pip install \
		"todd-ai[optional,dev,lint,doc,test] @ git+https://github.com/LutingWang/todd.git@$(1)"
	pipenv run pip uninstall -y opencv-python opencv-python-headless
	pipenv run pip install opencv-python-headless
endef

.PHONY: lint commit install_todd todd tb

lint:
	pipenv run pre-commit run -a

commit:
	pipenv run cz c

install_todd:
	$(call install_todd,$(current_todd_version))

todd:
	if [[ "$(latest_todd_version)" == "$(current_todd_version)" ]]; then \
		echo "No changes since last build."; \
		exit 1; \
	fi
	$(call install_todd,$(latest_todd_version))
	echo $(latest_todd_version) > .todd_version

tb:
	mkdir -p tensorboards
	for work_dir in work_dirs/*; do \
		name=$$(basename $${work_dir}); \
        tb_dir=$${work_dir}/tensorboard; \
        if [[ -d $${tb_dir} ]]; then \
            ln -sfT $$(realpath $${tb_dir}) tensorboards/$${name}; \
        fi; \
    done
	tensorboard --logdir tensorboards

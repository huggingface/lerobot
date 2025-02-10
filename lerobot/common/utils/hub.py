from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Type, TypeVar

from huggingface_hub import HfApi
from huggingface_hub.utils import validate_hf_hub_args

T = TypeVar("T", bound="HubMixin")


class HubMixin:
    """
    A Mixin containing the functionality to push an object to the hub.

    This is similar to huggingface_hub.ModelHubMixin but is lighter and makes less assumptions about its
    subclasses (in particular, the fact that it's not necessarily a model).

    The inheriting classes must implement '_save_pretrained' and 'from_pretrained'.
    """

    def save_pretrained(
        self,
        save_directory: str | Path,
        *,
        repo_id: str | None = None,
        push_to_hub: bool = False,
        card_kwargs: dict[str, Any] | None = None,
        **push_to_hub_kwargs,
    ) -> str | None:
        """
        Save object in local directory.

        Args:
            save_directory (`str` or `Path`):
                Path to directory in which the object will be saved.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your object to the Huggingface Hub after saving it.
            repo_id (`str`, *optional*):
                ID of your repository on the Hub. Used only if `push_to_hub=True`. Will default to the folder name if
                not provided.
            card_kwargs (`Dict[str, Any]`, *optional*):
                Additional arguments passed to the card template to customize the card.
            push_to_hub_kwargs:
                Additional key word arguments passed along to the [`~HubMixin.push_to_hub`] method.
        Returns:
            `str` or `None`: url of the commit on the Hub if `push_to_hub=True`, `None` otherwise.
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # save object (weights, files, etc.)
        self._save_pretrained(save_directory)

        # push to the Hub if required
        if push_to_hub:
            if repo_id is None:
                repo_id = save_directory.name  # Defaults to `save_directory` name
            return self.push_to_hub(repo_id=repo_id, card_kwargs=card_kwargs, **push_to_hub_kwargs)
        return None

    def _save_pretrained(self, save_directory: Path) -> None:
        """
        Overwrite this method in subclass to define how to save your object.

        Args:
            save_directory (`str` or `Path`):
                Path to directory in which the object files will be saved.
        """
        raise NotImplementedError

    @classmethod
    @validate_hf_hub_args
    def from_pretrained(
        cls: Type[T],
        pretrained_name_or_path: str | Path,
        *,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        **kwargs,
    ) -> T:
        """
        Download the object from the Huggingface Hub and instantiate it.

        Args:
            pretrained_name_or_path (`str`, `Path`):
                - Either the `repo_id` (string) of the object hosted on the Hub, e.g. `lerobot/diffusion_pusht`.
                - Or a path to a `directory` containing the object files saved using `.save_pretrained`,
                    e.g., `../path/to/my_model_directory/`.
            revision (`str`, *optional*):
                Revision on the Hub. Can be a branch name, a git tag or any commit id.
                Defaults to the latest commit on `main` branch.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether to force (re-)downloading the files from the Hub, overriding the existing cache.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on every request.
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. By default, it will use the token
                cached when running `huggingface-cli login`.
            cache_dir (`str`, `Path`, *optional*):
                Path to the folder where cached files are stored.
            local_files_only (`bool`, *optional*, defaults to `False`):
                If `True`, avoid downloading the file and return the path to the local cached file if it exists.
            kwargs (`Dict`, *optional*):
                Additional kwargs to pass to the object during initialization.
        """
        raise NotImplementedError

    @validate_hf_hub_args
    def push_to_hub(
        self,
        repo_id: str,
        *,
        commit_message: str | None = None,
        private: bool | None = None,
        token: str | None = None,
        branch: str | None = None,
        create_pr: bool | None = None,
        allow_patterns: list[str] | str | None = None,
        ignore_patterns: list[str] | str | None = None,
        delete_patterns: list[str] | str | None = None,
        card_kwargs: dict[str, Any] | None = None,
    ) -> str:
        """
        Upload model checkpoint to the Hub.

        Use `allow_patterns` and `ignore_patterns` to precisely filter which files should be pushed to the hub. Use
        `delete_patterns` to delete existing remote files in the same commit. See [`upload_folder`] reference for more
        details.

        Args:
            repo_id (`str`):
                ID of the repository to push to (example: `"username/my-model"`).
            commit_message (`str`, *optional*):
                Message to commit while pushing.
            private (`bool`, *optional*):
                Whether the repository created should be private.
                If `None` (default), the repo will be public unless the organization's default is private.
            token (`str`, *optional*):
                The token to use as HTTP bearer authorization for remote files. By default, it will use the token
                cached when running `huggingface-cli login`.
            branch (`str`, *optional*):
                The git branch on which to push the model. This defaults to `"main"`.
            create_pr (`boolean`, *optional*):
                Whether or not to create a Pull Request from `branch` with that commit. Defaults to `False`.
            allow_patterns (`List[str]` or `str`, *optional*):
                If provided, only files matching at least one pattern are pushed.
            ignore_patterns (`List[str]` or `str`, *optional*):
                If provided, files matching any of the patterns are not pushed.
            delete_patterns (`List[str]` or `str`, *optional*):
                If provided, remote files matching any of the patterns will be deleted from the repo.
            card_kwargs (`Dict[str, Any]`, *optional*):
                Additional arguments passed to the card template to customize the card.

        Returns:
            The url of the commit of your object in the given repository.
        """
        api = HfApi(token=token)
        repo_id = api.create_repo(repo_id=repo_id, private=private, exist_ok=True).repo_id

        if commit_message is None:
            if "Policy" in self.__class__.__name__:
                commit_message = "Upload policy"
            elif "Config" in self.__class__.__name__:
                commit_message = "Upload config"
            else:
                commit_message = f"Upload {self.__class__.__name__}"

        # Push the files to the repo in a single commit
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            saved_path = Path(tmp) / repo_id
            self.save_pretrained(saved_path, card_kwargs=card_kwargs)
            return api.upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=saved_path,
                commit_message=commit_message,
                revision=branch,
                create_pr=create_pr,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                delete_patterns=delete_patterns,
            )

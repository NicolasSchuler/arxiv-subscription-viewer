"""Shell completion script generators for bash, zsh, and fish."""

from __future__ import annotations

# Static completion scripts generated from the known CLI structure.
# No runtime dependency on the application — scripts are self-contained.

_BASH_SCRIPT = r"""# bash completion for arxiv-viewer
# Add to ~/.bashrc:  eval "$(arxiv-viewer completions bash)"

_arxiv_viewer() {
    local cur prev words cword
    _init_completion || return

    local commands="browse search dates completions config-path doctor"
    local global_opts="--debug --color --no-color --ascii --version -V --help -h"

    # Find the subcommand (skip global flags)
    local cmd=""
    local i
    for ((i=1; i < cword; i++)); do
        case "${words[i]}" in
            --color) ((i++)) ;;  # skip --color's argument
            --debug|--no-color|--ascii|--version|-V) ;;
            -*) ;;
            *) cmd="${words[i]}"; break ;;
        esac
    done

    if [[ -z "$cmd" ]]; then
        COMPREPLY=($(compgen -W "$commands $global_opts" -- "$cur"))
        return
    fi

    case "$cmd" in
        browse)
            COMPREPLY=($(compgen -W "-i --input --no-restore --date --help -h" -- "$cur"))
            ;;
        search)
            case "$prev" in
                --field)
                    COMPREPLY=($(compgen -W "all title author abstract" -- "$cur"))
                    return ;;
                --mode)
                    COMPREPLY=($(compgen -W "latest page" -- "$cur"))
                    return ;;
            esac
            COMPREPLY=($(compgen -W "--query --field --category --mode --max-results --help -h" -- "$cur"))
            ;;
        dates|config-path|doctor)
            COMPREPLY=($(compgen -W "--help -h" -- "$cur"))
            ;;
        completions)
            COMPREPLY=($(compgen -W "bash zsh fish" -- "$cur"))
            ;;
    esac
}

complete -F _arxiv_viewer arxiv-viewer
"""

_ZSH_SCRIPT = r"""#compdef arxiv-viewer
# zsh completion for arxiv-viewer
# Add to ~/.zshrc:  eval "$(arxiv-viewer completions zsh)"

_arxiv_viewer() {
    local -a commands
    commands=(
        'browse:Open local history or a local paper file'
        'search:Fetch startup papers from the arXiv API'
        'dates:List available local history dates and exit'
        'completions:Generate shell completion scripts'
        'config-path:Print the configuration file path'
        'doctor:Check environment and configuration health'
    )

    local -a global_opts
    global_opts=(
        '--debug[Enable debug logging]'
        '--color[Color output mode]:mode:(auto always never)'
        '--no-color[Disable terminal colors]'
        '--ascii[Use ASCII-only status icons]'
        '(-V --version)'{-V,--version}'[Show version]'
        '(-h --help)'{-h,--help}'[Show help]'
    )

    _arguments -C \
        $global_opts \
        '1:command:->command' \
        '*::arg:->args'

    case "$state" in
        command)
            _describe -t commands 'arxiv-viewer command' commands
            ;;
        args)
            case "$words[1]" in
                browse)
                    _arguments \
                        '(-i --input)'{-i,--input}'[Input file]:file:_files' \
                        '--no-restore[Start with fresh session]' \
                        '--date[Open specific history date]:date:' \
                        '(-h --help)'{-h,--help}'[Show help]'
                    ;;
                search)
                    _arguments \
                        '--query[Query text]:query:' \
                        '--field[Search field]:field:(all title author abstract)' \
                        '--category[Category filter]:category:' \
                        '--mode[Search mode]:mode:(latest page)' \
                        '--max-results[API page size]:number:' \
                        '(-h --help)'{-h,--help}'[Show help]'
                    ;;
                dates|config-path|doctor)
                    _arguments '(-h --help)'{-h,--help}'[Show help]'
                    ;;
                completions)
                    _arguments '1:shell:(bash zsh fish)'
                    ;;
            esac
            ;;
    esac
}

_arxiv_viewer "$@"
"""

_FISH_SCRIPT = r"""# fish completion for arxiv-viewer
# Add to ~/.config/fish/config.fish:  arxiv-viewer completions fish | source

# Disable file completions by default
complete -c arxiv-viewer -f

# Global options
complete -c arxiv-viewer -n '__fish_use_subcommand' -l debug -d 'Enable debug logging'
complete -c arxiv-viewer -n '__fish_use_subcommand' -l color -x -a 'auto always never' -d 'Color output mode'
complete -c arxiv-viewer -n '__fish_use_subcommand' -l no-color -d 'Disable terminal colors'
complete -c arxiv-viewer -n '__fish_use_subcommand' -l ascii -d 'Use ASCII-only status icons'
complete -c arxiv-viewer -n '__fish_use_subcommand' -s V -l version -d 'Show version'

# Subcommands
complete -c arxiv-viewer -n '__fish_use_subcommand' -a browse -d 'Open local history or a local paper file'
complete -c arxiv-viewer -n '__fish_use_subcommand' -a search -d 'Fetch startup papers from the arXiv API'
complete -c arxiv-viewer -n '__fish_use_subcommand' -a dates -d 'List available local history dates and exit'
complete -c arxiv-viewer -n '__fish_use_subcommand' -a completions -d 'Generate shell completion scripts'
complete -c arxiv-viewer -n '__fish_use_subcommand' -a config-path -d 'Print the configuration file path'
complete -c arxiv-viewer -n '__fish_use_subcommand' -a doctor -d 'Check environment and configuration health'

# browse options
complete -c arxiv-viewer -n '__fish_seen_subcommand_from browse' -s i -l input -r -d 'Input file'
complete -c arxiv-viewer -n '__fish_seen_subcommand_from browse' -l no-restore -d 'Start with fresh session'
complete -c arxiv-viewer -n '__fish_seen_subcommand_from browse' -l date -x -d 'Open specific history date (YYYY-MM-DD)'

# search options
complete -c arxiv-viewer -n '__fish_seen_subcommand_from search' -l query -x -d 'Query text'
complete -c arxiv-viewer -n '__fish_seen_subcommand_from search' -l field -x -a 'all title author abstract' -d 'Search field'
complete -c arxiv-viewer -n '__fish_seen_subcommand_from search' -l category -x -d 'Category filter (e.g. cs.AI)'
complete -c arxiv-viewer -n '__fish_seen_subcommand_from search' -l mode -x -a 'latest page' -d 'Search mode'
complete -c arxiv-viewer -n '__fish_seen_subcommand_from search' -l max-results -x -d 'API page size'

# completions options
complete -c arxiv-viewer -n '__fish_seen_subcommand_from completions' -a 'bash zsh fish' -d 'Shell type'
"""

SUPPORTED_SHELLS = ("bash", "zsh", "fish")

_SCRIPTS: dict[str, str] = {
    "bash": _BASH_SCRIPT,
    "zsh": _ZSH_SCRIPT,
    "fish": _FISH_SCRIPT,
}


def get_completion_script(shell: str) -> str:
    """Return the completion script for the given shell.

    Raises:
        ValueError: If the shell is not supported.
    """
    script = _SCRIPTS.get(shell)
    if script is None:
        supported = ", ".join(SUPPORTED_SHELLS)
        raise ValueError(f"Unsupported shell: {shell!r}. Supported shells: {supported}")
    return script.lstrip("\n")


__all__ = [
    "SUPPORTED_SHELLS",
    "get_completion_script",
]

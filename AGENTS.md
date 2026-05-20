# Claude Code 安装与项目接入流程

这份文档用于让新的 Agent 在一台相同用户环境的机器上，复现当前 Claude Code 配置。目标是完成以下结果：

- 安装 `claude` 命令。
- 配置 `~/.claude/settings.json` 的 `env`。
- 支持 provider 文件：`~/.claude/providers/<provider_name>.json`。
- 支持 `cc` 命令：
  - `cc` 等价于 `claude --dangerously-skip-permissions`。
  - `cc switch <provider_name>` 用 provider 文件整体替换 `settings.json` 里的 `env`。
  - `cc set <provider_name> token <token_key>` 只更新 provider 文件里的 `ANTHROPIC_AUTH_TOKEN`，不直接改 `settings.json`。
  - `cc next` 自动轮询切换到下一个 provider。
- 全局 Claude 技能目录只保留用户级全局技能。
- 当前项目 Claude 技能目录平铺链接项目技能和 OpenHarness 技能。
- 当前项目根目录 `CLAUDE.md` 软链接到 `AGENTS.md`。

## 1. 前置条件

确认系统有 Node.js、npm 和 jq：

```bash
node -v
npm -v
jq --version
```

如果缺少 `jq`，先用系统包管理器安装。后续脚本依赖 `jq` 修改 JSON。

## 2. 安装 Claude Code

```bash
npm install -g @anthropic-ai/claude-code
command -v claude
claude --version
```

期望结果：

- `command -v claude` 能输出一个可执行路径。
- `claude --version` 能输出 Claude Code 版本。

## 3. 创建 Claude 配置目录

```bash
mkdir -p ~/.claude/providers
mkdir -p ~/.local/bin
```

## 4. 创建 provider 文件

provider 文件是一个 JSON 字典。`cc switch <provider_name>` 会把这个字典直接写入 `~/.claude/settings.json` 的 `env` 键。

### 4.1 DeepSeek provider

写入 `~/.claude/providers/deepseek.json`：

```bash
cat > ~/.claude/providers/deepseek.json <<'JSON'
{
  "ANTHROPIC_BASE_URL": "https://api.deepseek.com/anthropic",
  "ANTHROPIC_AUTH_TOKEN": "<你的 DeepSeek API Key>",
  "ANTHROPIC_MODEL": "deepseek-v4-pro[1m]",
  "ANTHROPIC_DEFAULT_OPUS_MODEL": "deepseek-v4-pro[1m]",
  "ANTHROPIC_DEFAULT_SONNET_MODEL": "deepseek-v4-pro[1m]",
  "ANTHROPIC_DEFAULT_HAIKU_MODEL": "deepseek-v4-flash",
  "CLAUDE_CODE_SUBAGENT_MODEL": "deepseek-v4-flash",
  "CLAUDE_CODE_EFFORT_LEVEL": "max"
}
JSON
```

### 4.2 Kimi provider

写入 `~/.claude/providers/kimi.json`：

```bash
cat > ~/.claude/providers/kimi.json <<'JSON'
{
  "ANTHROPIC_BASE_URL": "https://api.kimi.com/coding/",
  "ANTHROPIC_AUTH_TOKEN": "${YOUR_MOONSHOT_API_KEY}",
  "ANTHROPIC_MODEL": "kimi-for-coding",
  "ANTHROPIC_DEFAULT_OPUS_MODEL": "kimi-for-coding",
  "ANTHROPIC_DEFAULT_SONNET_MODEL": "kimi-for-coding",
  "ANTHROPIC_DEFAULT_HAIKU_MODEL": "kimi-for-coding",
  "CLAUDE_CODE_SUBAGENT_MODEL": "kimi-for-coding",
  "ENABLE_TOOL_SEARCH": "true"
}
JSON
```

## 5. 初始化 settings.json

如果还没有 `~/.claude/settings.json`，可以先用 `deepseek` 初始化：

```bash
jq -n --argfile env ~/.claude/providers/deepseek.json '{env: $env}' > ~/.claude/settings.json
```

如果已有 `settings.json`，只替换其中的 `env`，保留其他设置：

```bash
tmp_file="$(mktemp)"
jq --argfile env ~/.claude/providers/deepseek.json '.env = $env' ~/.claude/settings.json > "$tmp_file"
mv "$tmp_file" ~/.claude/settings.json
```

## 6. 配置 cc 命令

系统里通常已经有 `/usr/bin/cc` 编译器，所以不能只创建 `~/.local/bin/cc`。为了让交互式 shell 优先识别 Claude 的 `cc`，需要把下面函数追加到 `~/.bashrc`。

```bash
cat >> ~/.bashrc <<'BASH'

cc() {
    local claude_bin
    claude_bin="$(command -v claude)"
    local claude_dir="${CLAUDE_HOME:-$HOME/.claude}"
    local providers_dir="$claude_dir/providers"
    local settings_file="$claude_dir/settings.json"
    local current_provider_file="$claude_dir/.current_provider"

    case "${1:-}" in
        switch)
            local provider_name provider_file tmp_file
            provider_name="${2:-}"
            if [[ -z "$provider_name" ]]; then
                echo "用法: cc switch <provider_name>" >&2
                return 1
            fi

            provider_file="$providers_dir/$provider_name.json"

            if [[ ! -f "$provider_file" ]]; then
                echo "找不到 provider: $provider_name" >&2
                return 1
            fi

            mkdir -p "$(dirname "$settings_file")"
            tmp_file="$(mktemp)"
            if [[ -f "$settings_file" ]]; then
                jq --argfile env "$provider_file" '.env = $env' "$settings_file" > "$tmp_file"
            else
                jq -n --argfile env "$provider_file" '{env: $env}' > "$tmp_file"
            fi
            mv "$tmp_file" "$settings_file"
            echo "$provider_name" > "$current_provider_file"
            echo "已切换到 $provider_name"
            ;;
        set)
            local provider_name field_name field_value provider_file tmp_file
            provider_name="${2:-}"
            field_name="${3:-}"
            field_value="${4:-}"
            if [[ -z "$provider_name" || -z "$field_name" || -z "$field_value" ]]; then
                echo "用法: cc set <provider_name> token <token_key>" >&2
                return 1
            fi

            if [[ "$field_name" != "token" ]]; then
                echo "当前只支持: cc set <provider_name> token <token_key>" >&2
                return 1
            fi

            provider_file="$providers_dir/$provider_name.json"

            if [[ ! -f "$provider_file" ]]; then
                echo "找不到 provider: $provider_name" >&2
                return 1
            fi

            tmp_file="$(mktemp)"
            jq --arg token "$field_value" '.ANTHROPIC_AUTH_TOKEN = $token' "$provider_file" > "$tmp_file"
            mv "$tmp_file" "$provider_file"
            echo "已更新 $provider_name 的 ANTHROPIC_AUTH_TOKEN"
            ;;
        next)
            local provider_files current_provider next_provider next_index
            mapfile -t provider_files < <(ls -1 "$providers_dir"/*.json 2>/dev/null | sort)

            if [[ ${#provider_files[@]} -eq 0 ]]; then
                echo "找不到任何 provider 文件" >&2
                return 1
            fi

            current_provider=""
            if [[ -f "$current_provider_file" ]]; then
                current_provider="$(cat "$current_provider_file" 2>/dev/null)"
            fi

            if [[ -z "$current_provider" ]]; then
                next_index=0
            else
                next_index=-1
                for i in "${!provider_files[@]}"; do
                    if [[ "$(basename "${provider_files[$i]}" .json)" == "$current_provider" ]]; then
                        next_index=$(( (i + 1) % ${#provider_files[@]} ))
                        break
                    fi
                done
                if [[ $next_index -eq -1 ]]; then
                    next_index=0
                fi
            fi

            next_provider="$(basename "${provider_files[$next_index]}" .json)"
            cc switch "$next_provider"
            ;;
        "")
            exec "$claude_bin" --dangerously-skip-permissions
            ;;
        *)
            exec "$claude_bin" --dangerously-skip-permissions "$@"
            ;;
    esac
}
BASH
```

让当前 shell 生效：

```bash
source ~/.bashrc
type cc
```

期望 `type cc` 显示 `cc is a function`。

## 7. 使用 provider 命令

设置某个 provider 的令牌，只修改 provider 文件：

```bash
cc set deepseek token '<真实 DeepSeek API Key>'
cc set kimi token '<真实 Moonshot API Key>'
```

切换当前 Claude 环境：

```bash
cc switch deepseek
cc switch kimi
```

轮询切换到下一个 provider：

```bash
cc next
```

注意：

- `cc set <provider_name> token <token_key>` 不会自动更新 `settings.json`。
- 如果已经切换到某个 provider 后又更新了该 provider 的 token，需要再执行一次 `cc switch <provider_name>` 才会把新 token 写入当前 `settings.json`。

## 8. 配置技能软链接

Claude 只能识别平铺的 `skills/<skill_name>/SKILL.md` 结构。不要把 `.agents/skills` 这种嵌套目录整体链接过去。

## 链接 CLAUDE.md

在项目根目录执行：

```bash
ln -sfn AGENTS.md CLAUDE.md
ls -l CLAUDE.md
```

期望结果：

```text
CLAUDE.md -> AGENTS.md
```

comment from gpt

# AscendC 动态库加载缺符号：2026-11-16

证据：[本次完整server日志](https://github.com/ava131/GDN-RecoverSSM-Ascend/issues/4#issuecomment-6905293595)。修复仍在op-dev，main不改。

## 这次看哪一条

真正阻断启动的是01:51:48的底层OSError：

```text
libgdn_recoverssm_ac.so: undefined symbol:
_ZN16platform_ascendc22PlatformAscendCManager15platformInitMtxE
```

解码后是 `platform_ascendc::PlatformAscendCManager::platformInitMtx`，属于Host平台信息API的静态成员。

这次已经生成动态库，失败在 `torch.ops.load_library → ctypes.CDLL → dlopen`。上次PTR宏的编译阻断已过去，但算子自检尚未开始。前面的模型AOT cache加载warning触发了原生重新编译，不能把它当作本次最终失败的根因。

不需要再找driver/TRS日志，也不需要调整batch、图参数或async来定位这个Host加载问题。

## 已确认的构建遗漏

我们的ops.cpp使用PlatformAscendCManager查询AIV核数与UB，但CMake仅链接platform，漏掉了tiling_api。

[CANN官方文档](https://asc.gitcode.com/api/Utils-API/platform_info/PlatformAscendCManager.html)明确要求同时链接 `tiling_api` 和 `platform`。这是本插件的构建依赖缺口，不应要求用户通过更改全局库路径、预加载库或更换CANN版本绕过。

修复内容：

1. 显式查找当前GDN_CANN_ROOT及其架构目录下的tiling_api和platform，以绝对路径链接，避免构建时误选其他SDK的库。
2. 为本插件的链接启用 `--no-undefined`，让插件对象的未解析符号在构建阶段失败，并进入已有compiler首错日志。不启用对全部间接共享库的严格未解析检查，避免将由Python进程提供的间接符号误当作插件缺依赖。
3. 保留Torch包装异常的内部cause链，server最后的AscendCBuildError会直接携带缺失符号，标识为 `GDN_ASCENDC_LOAD_FAILURE`，不再只剩“Could not load this library”。
4. 设备kernel、精度、物理页合同及启动参数保持当前实现。

## 验证与边界

新增检查覆盖真实CMake配置生成的链接命令：两个直接依赖必须来自指定SDK，必须有未定义符号检查，缺少依赖时配置失败；另有Torch包装OSError的缺符号信息保留检查。

配置测试用CPU编译器、模拟Torch配置和SDK文件，只验证构建配方，不冒充真实CANN链接/加载。开发机无NPU，本次不能宣称完整数值自检或图运行已经通过。若目标环境中仍存在SDK混用或其他未解析符号，修复后的构建/加载日志将给出新的具体错误。

交付前完整CPU套件159项通过（无跳过），其中4项实际运行CMake配置与链接命令检查。修改文件的关键lint、格式及diff检查通过。

## 如何更新

原仓库中执行：

```bash
git pull --ff-only origin op-dev
```

沿用实际Replay启动命令重启。新的CMake源码参与构建fingerprint，会自动重新构建；无需清缓存、增加开关或做二分实验。

当前已有日志足以修正上述缺依赖问题，不要求用户重新跑命令取同样的堆栈。


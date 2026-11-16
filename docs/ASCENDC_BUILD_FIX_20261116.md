comment from gpt

# AscendC 首次构建失败修复（2026-11-16）

分支：`op-dev`。对应现场实现基线：`8b3c51b`。证据为 [#4 的完整 build.log](https://github.com/ava131/GDN-RecoverSSM-Ascend/issues/4#issuecomment-6904987132)。

## 确定的根因

CMake 已完成配置和生成，BiSheng 在编译 `ops.cpp.o` 时拒绝：

```text
error: reinterpret_cast from 'void *' to '__gm__ uint8_t *' is not allowed
#define PTR(tensor) reinterpret_cast<GM_ADDR>((tensor).data_ptr())
```

Tensor.data_ptr()返回普通void*，GM_ADDR带有AscendC的GM地址空间限定。C++ reinterpret_cast不允许这样跨地址空间转换。verify的12个Tensor指针和fold的18个Tensor指针共用这个宏，因此一个错误重复触发多次，最后到达编译器20个error限制。这是我在Host发射适配代码里的错误。

日志里的kineto静态库缺失warning、Torch-NPU默认move构造被删除warning，没有构成本次报错的阻断点。没有证据需要修改用户启动参数、架构选项、Torch或CANN版本。

此前普通C++ Host probe将GM_ADDR模拟成普通指针，丢失了地址空间限定，因此未检出该问题。这次新增检查保留了该限定。

## 修改内容

1. Host的PTR统一使用AscendC fast-launch支持的显式GM_ADDR转换：

   ```cpp
   #define PTR(tensor) ((GM_ADDR)((tensor).data_ptr()))
   ```

   这一次修改覆盖verify和fold的全部30个Host指针参数。不会分配或复制Tensor。

2. 设备头中的同GM地址空间类型转换、整数地址表到GM指针的转换保持原形式。它们不具有这次generic pointer跨地址空间的问题条件；已单独做真实address-space类型检查。

3. 构建失败时，异常自动附带 `GDN_ASCENDC_BUILD_FAILURE`、失败阶段、首个编译诊断、上下文和日志尾部，直接进入server日志。完整build.log仍保留；首错优先显示，长模板展开不会把首错挤出截断窗口。

4. CMake构建启用内部verbose日志，以便失败时保留真实编译/链接命令。没有新增用户开关。

5. 构建recipe更新，且C++源码内容参与fingerprint。更新代码后自动使用新缓存键重编，不需要用户删除旧缓存。

## 验证

- Clang使用真实 `address_space(1)` 类型编译：旧reinterpret_cast负例产生与现场同类错误；当前真实PTR宏通过。
- 同一编译检查还覆盖设备侧同GM类型转换与int64地址转GM指针。
- 构建异常回归检查覆盖compile/link阶段、长模板日志、首错不被截断、ANSI颜色、非法编码及日志不可读。
- 完整CPU测试：**154 passed**。
- 修改文件的关键Ruff、格式和git diff检查通过。
- 本机没有CANN/NPU；以上不是完整BiSheng设备编译、算子数值或图运行通过的证明。现场编译器因error limit停止，修复已知首错后仍可能暴露后续设备编译问题，不能提前宣称整条链路已通过。

## 更新方式

在原环境、原仓库中：

```bash
git pull --ff-only origin op-dev
```

然后沿用原Replay启动命令重启。若之前已按说明editable安装，代码会直接更新；无需新增参数，也无需手工清理缓存。数值自检和READY保护继续完整执行。

关注：

```bash
grep -E 'GDN_ASCENDC|GDN_READY' server.log
```

如果还有构建错误，server日志中的 `GDN_ASCENDC_BUILD_FAILURE` 会直接显示首错，避免再次只有CMake退出码。


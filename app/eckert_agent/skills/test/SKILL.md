---
name: api-design-principles
description: 精通 REST 与 GraphQL API 设计原则，打造直观易用、可扩展且易于维护的 API，为开发者带来优质使用体验。适用于新 API 设计、API 规格评审或 API 设计标准制定等场景。
---

# API 设计原则
精通 REST 与 GraphQL API 设计原则，打造直观易用、可扩展且易于维护的 API，为开发者带来优质使用体验，同时确保 API 能够长期稳定运行。

## 适用场景
- 设计全新的 REST 或 GraphQL API
- 重构现有 API，提升其易用性
- 为团队制定统一的 API 设计规范
- 开发前评审 API 规格说明书
- API 架构模式迁移（如从 REST 迁移至 GraphQL 等）
- 编写面向开发者的友好型 API 文档
- 针对特定场景优化 API（如移动端适配、第三方集成等）

## 不适用场景
- 仅需特定框架的 API 实现指导
- 仅开展基础设施相关工作，不涉及 API 契约定义
- 无法变更或版本化公共接口

## 设计步骤
1.  明确 API 使用者、核心用例及各项约束条件
2.  选定 API 架构风格，并完成资源或数据类型的建模
3.  制定错误处理、版本控制、分页机制及认证授权策略
4.  通过示例验证设计合理性，并审核整体一致性

详细的设计模式、检查清单及模板，请参考 `resources/implementation-playbook.md` 文件。

## 参考资源
- `resources/implementation-playbook.md`：内含详细设计模式、检查清单及模板

---

### 翻译说明
1.  **术语统一**
    - REST/GraphQL 保留英文，为技术领域通用术语
    - `API design principles` 译为「API 设计原则」，符合行业惯例
    - `resources or types` 译为「资源或数据类型」，兼顾 REST（资源）与 GraphQL（类型）的特性
    - `pagination` 译为「分页机制」，`auth strategy` 译为「认证授权策略」，精准对应技术概念
2.  **句式优化**
    - 英文长句拆解为符合中文表达习惯的短句（如开篇句）
    - 被动语态转主动语态（如 `Validate with examples` 译为「通过示例验证」）
    - 祈使句保持简洁有力（如设计步骤部分）
3.  **语境适配**
    - `delight developers` 译为「为开发者带来优质使用体验」，避免直译「取悦开发者」的生硬感
    - `stand the test of time` 译为「长期稳定运行」，贴合 API 设计的实用性需求
    - `public interfaces` 译为「公共接口」，为软件工程标准译法
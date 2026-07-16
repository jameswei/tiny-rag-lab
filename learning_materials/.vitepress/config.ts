import { defineConfig } from "vitepress";

type Guide = { text: string; slug: string };

const englishGuides: Guide[] = [
  { text: "The RAG Data Flow", slug: "the-rag-data-flow" },
  { text: "The Indexing Plane", slug: "the-indexing-plane" },
  { text: "Retrieval and Generation", slug: "retrieval-and-generation" },
  { text: "Persistence and Testing", slug: "persistence-and-testing" },
  { text: "Retrieval Mechanics", slug: "retrieval-mechanics" },
  { text: "Reranking", slug: "reranking" },
  { text: "Evaluating Retrieval", slug: "evaluating-retrieval" },
  { text: "Observability and Debugging", slug: "observability-and-debugging" },
  { text: "RAG Failure Lab", slug: "rag-failure-lab" },
  { text: "Answer Quality Judging", slug: "answer-quality-judging" },
  { text: "Context Budget and Structured Answers", slug: "context-budget-and-structured-answers" },
  { text: "Structural and Semantic Chunking", slug: "structural-and-semantic-chunking" },
];

const chineseGuides: Guide[] = [
  { text: "RAG 数据流", slug: "the-rag-data-flow" },
  { text: "索引平面", slug: "the-indexing-plane" },
  { text: "检索与生成", slug: "retrieval-and-generation" },
  { text: "持久化与测试", slug: "persistence-and-testing" },
  { text: "检索机制", slug: "retrieval-mechanics" },
  { text: "重排序", slug: "reranking" },
  { text: "评估检索质量", slug: "evaluating-retrieval" },
  { text: "可观测性与调试", slug: "observability-and-debugging" },
  { text: "RAG 失败实验室", slug: "rag-failure-lab" },
  { text: "答案质量评判", slug: "answer-quality-judging" },
  { text: "上下文预算与结构化答案", slug: "context-budget-and-structured-answers" },
  { text: "结构化与语义分块", slug: "structural-and-semantic-chunking" },
];

function sidebar(language: "en" | "zh", title: string, guides: Guide[]) {
  return [
    {
      text: title,
      items: [
        { text: language === "en" ? "Learning roadmap" : "学习路线图", link: `/${language}/learning-roadmap.html` },
        ...guides.map((guide) => ({ text: guide.text, link: `/${language}/${guide.slug}.html` })),
      ],
    },
  ];
}

export default defineConfig({
  base: "/docs/",
  appearance: false,
  cleanUrls: false,
  title: "tiny-rag-lab Learning Guides",
  description: "Concept-focused guides for understanding the inspectable classic RAG pipeline.",
  head: [["link", { rel: "icon", type: "image/svg+xml", href: "/docs/favicon.svg" }]],
  locales: {
    en: {
      label: "English",
      lang: "en",
      link: "/en/",
      title: "tiny-rag-lab Learning Guides",
      description: "Concept-focused guides for understanding the inspectable classic RAG pipeline.",
      themeConfig: {
        siteTitle: "Learning Guides",
        nav: [
          { text: "Learning roadmap", link: "/en/learning-roadmap.html" },
          { component: "LabReturnLink", props: { label: "Back to lab" } },
        ],
        sidebar: { "/en/": sidebar("en", "Classic RAG", englishGuides) },
        outline: { label: "On this page", level: "deep" },
        docFooter: { prev: "Previous guide", next: "Next guide" },
        langMenuLabel: "Change language",
        sidebarMenuLabel: "Guide menu",
        returnToTopLabel: "Return to top",
      },
    },
    zh: {
      label: "简体中文",
      lang: "zh-CN",
      link: "/zh/",
      title: "tiny-rag-lab 学习指南",
      description: "通过概念指南理解可检查的经典 RAG 流程。",
      themeConfig: {
        siteTitle: "学习指南",
        nav: [
          { text: "学习路线图", link: "/zh/learning-roadmap.html" },
          { component: "LabReturnLink", props: { label: "返回实验室" } },
        ],
        sidebar: { "/zh/": sidebar("zh", "经典 RAG", chineseGuides) },
        outline: { label: "本页内容", level: "deep" },
        docFooter: { prev: "上一篇", next: "下一篇" },
        langMenuLabel: "切换语言",
        sidebarMenuLabel: "指南目录",
        returnToTopLabel: "返回顶部",
      },
    },
  },
  themeConfig: {
    i18nRouting: true,
    search: {
      provider: "local",
      options: {
        locales: {
          zh: {
            translations: {
              button: { buttonText: "搜索", buttonAriaLabel: "搜索学习指南" },
              modal: {
                displayDetails: "显示详细列表",
                resetButtonTitle: "清除搜索",
                backButtonTitle: "关闭搜索",
                noResultsText: "没有找到相关内容",
                footer: {
                  selectText: "选择",
                  selectKeyAriaLabel: "回车",
                  navigateText: "导航",
                  navigateUpKeyAriaLabel: "上箭头",
                  navigateDownKeyAriaLabel: "下箭头",
                  closeText: "关闭",
                  closeKeyAriaLabel: "Esc",
                },
              },
            },
          },
        },
      },
    },
    socialLinks: [
      { icon: "github", link: "https://github.com/jameswei/tiny-rag-lab", ariaLabel: "tiny-rag-lab on GitHub" },
    ],
  },
});

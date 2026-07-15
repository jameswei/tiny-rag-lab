<script setup lang="ts">
import { onMounted } from "vue";

const languageKey = "tiny-rag-lab-lang";

function saveLanguage(language: "en" | "zh") {
  window.localStorage.setItem(languageKey, language);
}

onMounted(() => {
  const saved = window.localStorage.getItem(languageKey);
  const detected = window.navigator.languages.some((language) => language.toLowerCase().startsWith("zh")) ? "zh" : "en";
  const language = saved === "en" || saved === "zh" ? saved : detected;
  window.location.replace(`/docs/${language}/learning-roadmap.html`);
});
</script>

<template>
  <main class="language-entry">
    <span class="entry-kicker">tiny-rag-lab</span>
    <h1>Learning Guides · 学习指南</h1>
    <p>Choose a language to read the local guides. 选择学习指南的语言。</p>
    <div class="language-choices">
      <a href="/docs/en/learning-roadmap.html" @click="saveLanguage('en')">English</a>
      <a href="/docs/zh/learning-roadmap.html" @click="saveLanguage('zh')">简体中文</a>
    </div>
  </main>
</template>

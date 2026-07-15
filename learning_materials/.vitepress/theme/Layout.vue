<script setup lang="ts">
import { inBrowser, useData, useRoute } from "vitepress";
import DefaultTheme from "vitepress/theme";
import { computed, watchEffect } from "vue";

const { lang } = useData();
const route = useRoute();
const isChinese = computed(() => lang.value.startsWith("zh"));

function returnToLab() {
  window.location.assign(new URL("/", window.location.origin).href);
}

watchEffect(() => {
  if (!inBrowser || route.path === "/") return;
  window.localStorage.setItem("tiny-rag-lab-lang", lang.value.startsWith("zh") ? "zh" : "en");
});
</script>

<template>
  <DefaultTheme.Layout>
    <template #layout-bottom>
      <footer class="learning-footer">
        <div>
          <strong>tiny-rag-lab</strong>
          <span>{{ isChinese ? "由 James Wei 创建" : "Created by James Wei" }}</span>
        </div>
        <div>
          <a href="/" @click.prevent="returnToLab">{{ isChinese ? "返回实验室" : "Back to lab" }}</a>
          <a href="https://github.com/jameswei/tiny-rag-lab">GitHub</a>
        </div>
      </footer>
    </template>
  </DefaultTheme.Layout>
</template>

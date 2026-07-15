import DefaultTheme from "vitepress/theme";
import type { Theme } from "vitepress";
import LabReturnLink from "./LabReturnLink.vue";
import LanguageEntry from "./LanguageEntry.vue";
import Layout from "./Layout.vue";
import "./custom.css";

export default {
  extends: DefaultTheme,
  Layout,
  enhanceApp({ app }) {
    app.component("LabReturnLink", LabReturnLink);
    app.component("LanguageEntry", LanguageEntry);
  },
} satisfies Theme;

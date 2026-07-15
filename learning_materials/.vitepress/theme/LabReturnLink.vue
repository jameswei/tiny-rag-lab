<script setup lang="ts">
defineProps<{
  label: string;
  screenMenu?: boolean;
}>();

function returnToLab() {
  window.location.assign(new URL("/", window.location.origin).href);
}
</script>

<template>
  <a
    :class="['lab-return-nav', { 'screen-menu': screenMenu }]"
    href="/"
    @click.prevent="returnToLab"
  >
    {{ label }}
  </a>
</template>

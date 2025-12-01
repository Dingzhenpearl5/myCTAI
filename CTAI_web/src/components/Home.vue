<template>
  <div class="home">
    <app-header></app-header>
    <app-content></app-content>
    <app-footer></app-footer>
  </div>
</template>

<script>
import Header from './Header'
import Footer from './Footer'
import Content from './Content'

export default {
  name: "Home",
  components:{
    "app-header":Header,
    "app-footer":Footer,
    "app-content":Content
  }
};
</script>

<style scoped>
</style>

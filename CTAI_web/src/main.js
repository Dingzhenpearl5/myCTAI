// The Vue build version to load with the `import` command
// (runtime-only or standalone) has been set in webpack.base.conf with an alias.
import Vue from 'vue'
import App from './App'
import Home from './components/Home'
import Login from './components/Login'
import VueRouter from 'vue-router'
import axios from 'axios'
import Element from 'element-ui'
import echarts from "echarts";

Vue.prototype.$echarts = echarts;
import '../node_modules/element-ui/lib/theme-chalk/index.css'
import '../src/assets/style.css'
import './theme/index.css'

Vue.use(Element)
Vue.config.productionTip = false
Vue.use(VueRouter)
Vue.prototype.$http = axios

const router = new VueRouter({
    routes: [
        { path: '/', redirect: '/login' },
        { path: '/login', component: Login, meta: { title: "登录 - 肿瘤辅助诊断系统" } },
        { path: '/home', component: Home, meta: { title: "首页 - 肿瘤辅助诊断系统" } },
        // 兼容旧路由，重定向到home
        { path: "/App", redirect: '/home' },
    ],
    mode: "history"
})

// 路由守卫，设置标题
router.beforeEach((to, from, next) => {
  if (to.meta.title) {
    document.title = to.meta.title;
  }
  next();
});

// // 全局注册组件
Vue.component("App", App);

/* eslint-disable no-new */
new Vue({
    el: '#app',
    router,
    render: h => h(App)
})

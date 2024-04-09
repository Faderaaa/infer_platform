const welcome = httpVueLoader('/templates/component/welcome.vue')

const routes = [
    {path: '/welcome', component: welcome}
]

const router = new VueRouter({
  routes
})
Vue.use(httpVueLoader);
new Vue({
    el: '#app',
    router,
    data: function () {
        return{
            pageItem:"这里是主页的面包屑导航哦~"
        }},
    methods: {},
    delimiters: ["${", "}"]
})
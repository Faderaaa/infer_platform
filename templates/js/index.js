new Vue({
        el: '#app',
        data: function () {
            return {
                form: {
                    password: "",
                    username: "",
                },
                checked: false,
                rules: {
                    username: [
                        {required: true, message: "请输入用户名", trigger: "blur"},
                        {max: 10, message: "不能大于10个字符", trigger: "blur"},
                    ],
                    password: [
                        {required: true, message: "请输入密码", trigger: "blur"},
                        {max: 10, message: "不能大于10个字符", trigger: "blur"},
                    ],
                },
                systemName:"AIOT1206推理平台"
            }
        },
        mounted() {
            if (localStorage.getItem("news")) {
                this.form = JSON.parse(localStorage.getItem("news"))
                this.checked = true
            }
        },
        methods: {
            login(form) {
                this.$refs[form].validate((valid) => {
                    var that = this;
                    if (valid) {
                        // TODO 编写实际的登录校验
                        var xhr = new XMLHttpRequest();
                        xhr.open('POST', '/login', true);
                        xhr.setRequestHeader('Content-Type', 'application/json');
                        xhr.onreadystatechange = function() {
                        if (xhr.readyState === XMLHttpRequest.DONE && xhr.status === 200) {
                            let res = JSON.parse(xhr.response)
                            if (res['code'] === 20000){
                                that.$message({
                                        message: res['message'],
                                        type: "success",
                                        showClose: true,
                                });
                            location.replace("/home");
                            }else {
                                that.$message({
                                        message: res['message'],
                                        type: "error",
                                        showClose: true,
                                });
                            }
                            }
                        }
                        xhr.send((JSON.stringify(this.form)));
                    } else {
                        return false;
                    }
                });
            },
            remenber(data) {
                this.checked = data
                if (this.checked) {
                    localStorage.setItem("news", JSON.stringify(this.form))
                } else {
                    localStorage.removeItem("news")
                }
            },
            forgetpas() {
                this.$message({
                    type: "info",
                    message: "功能尚未开发额😥",
                    showClose: true
                })
            }
        },
        delimiters: ["${", "}"]
    })
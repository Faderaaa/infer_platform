<template>
  <div class="welcomeBody">
    <p>这是视频流展示demo（同时只能查看一个）.
    </p>
    <div class="inputDemo">
      <div class="inputItem" >rstpUrl:<el-input style="width: 50%" v-model="rstpUrl" placeholder="请输入rstpUrl"></el-input></div>
      <div class="inputItem" >模型名称:
        <el-select v-model="modelName" placeholder="请选择">
            <el-option v-for="item in options" :key="item.value" :label="item.label" :value="item.value"></el-option>
        </el-select>
      </div>
      <el-button @click="submit">查看</el-button>
    </div>
  </div>
</template>

<script>
module.exports = {
  data: function() {
    return {
      rstpUrl: "rtsp://admin:123456@192.168.31.68:554/ch01.264",
      modelName: "default",
      showInfer:false,
      options: [{
          value: 'default',
          label: '原始视频'
        },{
          value: 'yolov5s_bck.hef',
          label: 'yolov5s_bck'
        }]
    };
  },
   methods: {
    submit:function () {
      console.log(this.rstpUrl);
      window.open("/viewInfer?modelName=" + this.modelName + "&&rstpUrl=" + this.rstpUrl)
    },
   }
};
</script>

<style scoped>
  .welcomeBody{
    display: flex;
    align-items: center;
    flex-direction: column;
  }
  p {
    font-size: 2em;
    text-align: center;
  }
  .inputDemo{
    display: flex;
    flex-direction: column;
    width: 80%;
    gap: 20px;
  }
  .inputItem{
    display: flex;
    align-items: center;
    flex-direction: row;
    gap: 30px;
  }
</style>
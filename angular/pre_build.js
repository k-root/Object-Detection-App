console.log("pre-build");
var fs = require("fs");
var serviceFile = "./src/app/api-service/api.service.ts";
fs.readFile(serviceFile, "utf8", function(err, data) {
  if (err) {
    return console.log(err);
  }
  var result = data.replace(
    /BASE_URL: string = '([^]*)';/g,
    "BASE_URL: string = '';"
  );
  fs.writeFile(serviceFile, result, "utf8", function(err) {
    if (err) return console.log(err);
  });
});
console.log(`Replaced BASE_URL: string = 'http://127.0.0.1:8080'; in ${serviceFile} with BASE_URL: string = '';`);

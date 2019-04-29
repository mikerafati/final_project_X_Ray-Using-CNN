var tableData = data;

d3.select("tbody").selectAll().data(tableData).enter().append('tr').html(d=>{
    return `<td>${d.datetime}</td>
    <td>${d.city}</td>
    <td>${d.state}</td>
    <td>${d.country}</td>
    <td>${d.shape}</td>
    <td>${d.durationMinutes}</td>
    <td>${d.comments}</td>`
})


// Select the submit button
var submit = d3.select("#filter-btn");

submit.on("click", function() {

  // Prevent the page from refreshing
  d3.event.preventDefault();

  // Select the input element and get the raw HTML node
  var inputElement = d3.select("#datetime");

  // Get the value property of the input element
  var inputValue = inputElement.property("value");

  console.log(inputValue);
  console.log(tableData);

  var filteredData = tableData.filter(sighting => sighting.datetime === inputValue);

  console.log(filteredData)
  results(filteredData)
});



function results(data){
    var selection = d3.select("tbody").selectAll('tr').data(data);
    selection.enter().append("tr").merge(selection).html(d=>{
        return `<td>${d.datetime}</td>
        <td>${d.city}</td>
        <td>${d.state}</td>
        <td>${d.country}</td>
        <td>${d.shape}</td>
        <td>${d.durationMinutes}</td>
        <td>${d.comments}</td>`
    })
    selection.exit().remove();
}

